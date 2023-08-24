import argparse
import keyboard
import mediapipe as mp
import cv2
import numpy as np
from scipy.spatial import distance as dist
from math import atan, atan2, pi, degrees
from datetime import datetime

DEFAULT_LANDMARKS_STYLE = mp.solutions.drawing_styles.get_default_pose_landmarks_style()
DEFAULT_HAND_CONNECTIONS_STYLE = mp.solutions.drawing_styles.get_default_hand_connections_style()

# Optionally record the video feed to a timestamped AVI in the current directory
RECORDING_FILENAME = str(datetime.now()).replace('.','').replace(':','') + '.avi'
FPS = 10

VISIBILITY_THRESHOLD = .8 # amount of certainty that a body landmark is visible
STRAIGHT_LIMB_MARGIN = 20 # degrees from 180
EXTENDED_LIMB_MARGIN = .8 # lower limb length as fraction of upper limb

LEG_LIFT_MIN = -30 # degrees below horizontal

ARM_CROSSED_RATIO = 2 # max distance from wrist to opposite elbow, relative to mouth width

MOUTH_COVER_THRESHOLD = .03 # hands over mouth max distance error out of 1

SQUAT_THRESHOLD = .1 # max hip-to-knee vertical distance

JUMP_THRESHOLD = .0001

LEG_ARROW_ANGLE = 18 # degrees from vertical standing; should be divisor of 90

FINGER_MOUTH_RATIO = 1.5 # open hand relative to mouth width

# R side: 90 top to 0 right to -90 bottom
# L side: 90 top to 180 left to 269... -> -90 bottom
SEMAPHORES = {
  (-90, -45): {'a': "a", 'n': "1"},
  (-90, 0): {'a': "b", 'n': "2"},
  (-90, 45): {'a': "c", 'n': "3"},
  (-90, 90): {'a': "d", 'n': "4"},
  (135, -90): {'a': "e", 'n': "5"},
  (180, -90): {'a': "f", 'n': "6"},
  (225, -90): {'a': "g", 'n': "7"},
  (-45, 0): {'a': "h", 'n': "8"},
  (-45, 45): {'a': "i", 'n': "9"},
  (180, 90): {'a': "j", 'n': "capslock"},
  (90, -45): {'a': "k", 'n': "0"},
  (135, -45): {'a': "l", 'n': "\\"},
  (180, -45): {'a': "m", 'n': "["},
  (225, -45): {'a': "n", 'n': "]"},
  (0, 45): {'a': "o", 'n': ","},
  (90, 0): {'a': "p", 'n': ";"},
  (135, 0): {'a': "q", 'n': "="},
  (180, 0): {'a': "r", 'n': "-"},
  (225, 0): {'a': "s", 'n': "."},
  (90, 45): {'a': "t", 'n': "`"},
  (135, 45): {'a': "u", 'n': "/"},
  (225, 90): {'a': "v", 'n': '"'},
  (135, 180): {'a': "w"},
  (135, 225): {'a': "x", 'n': ""}, # clear last signal
  (180, 45): {'a': "y"},
  (180, 225): {'a': "z"},
  (90, 90): {'a': "space", 'n': "enter"},
  (135, 90): {'a': "tab"}, # custom "numerals" replacement
  (225, 45): {'a': "escape"}, # custom "cancel" replacement
}

leg_arrow_angles = {
  (-90, -90 + LEG_ARROW_ANGLE): "right",
  (-90, -90 + 2*LEG_ARROW_ANGLE): "up",
  (270 - LEG_ARROW_ANGLE, -90): "left",
  (270 - 2*LEG_ARROW_ANGLE, -90): "down",
}

FRAME_HISTORY = 8 # pose history is compared against FRAME_HISTORY recent frames
HALF_HISTORY = int(FRAME_HISTORY/2)
QUARTER_HISTORY = int(FRAME_HISTORY/4)

empty_frame = {
  'hipL_y': 0,
  'hipR_y': 0,
  'hips_dy': 0,
  'dxL_thrust_hipL': 0,
  'dxL_thrust_hipR': 0,
  'dxR_thrust_hipL': 0,
  'dxR_thrust_hipR': 0,
  'signed': False,
}
last_frames = FRAME_HISTORY*[empty_frame.copy()]

frame_midpoint = (0,0)

current_semaphore = ''
last_keys = []

def get_vector_angle(a, b):
  radian = np.arccos(np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b)))
  degress = np.degrees(radian)
  return degress

def get_vector_from_ldmks(ldmk_tail, landmk_head):
  ldmk_vec = np.array([landmk_head['x'], landmk_head['y']]) - np.array([ldmk_tail['x'], ldmk_tail['y']])
  return ldmk_vec

def get_distance_from_ldmks(ldmk_tail, landmk_head, X=1,Y=1):
  ldmk_dis = np.linalg.norm([landmk_head['x']*X, landmk_head['y']*Y] - np.array([ldmk_tail['x']*X, ldmk_tail['y']*Y]))
  return ldmk_dis

def get_mid_point_from_ldmks(ldmk_tail, landmk_head, X=1,Y=1):
  mid_point = ([landmk_head['x']*X, landmk_head['y']*Y] + np.array([ldmk_tail['x']*X, ldmk_tail['y']*Y]))/2
  return mid_point

def is_thumb_red_heart(ldmk_3, ldmk_4, ldmk_6, ldmk_8):
  vector_thumb = get_vector_from_ldmks(ldmk_3, ldmk_4)
  vector_figure = get_vector_from_ldmks(ldmk_6, ldmk_8)
  angle = get_vector_angle(vector_thumb, vector_figure)
  distance = get_distance_from_ldmks(ldmk_8, ldmk_4, X=1, Y=1)
  flag = (angle > 0.5) and (angle < 45) and (distance < 0.15)
  print('angle = ', angle)
  return flag

def get_angle(a, b, c):
  ang = degrees(atan2(c['y']-b['y'], c['x']-b['x']) - atan2(a['y']-b['y'], a['x']-b['x']))
  return ang + 360 if ang < 0 else ang

def is_missing(part):
  return any(joint['visibility'] < VISIBILITY_THRESHOLD for joint in part)

def is_limb_pointing(upper, mid, lower):
  if is_missing([upper, mid, lower]):
    return False
  limb_angle = get_angle(upper, mid, lower)
  is_in_line = abs(180 - limb_angle) < STRAIGHT_LIMB_MARGIN
  if is_in_line:
    upper_length = dist.euclidean([upper['x'], upper['y']], [mid['x'], mid['y']])
    lower_length = dist.euclidean([lower['x'], lower['y']], [mid['x'], mid['y']])
    is_extended = lower_length > EXTENDED_LIMB_MARGIN * upper_length
    return is_extended
  return False

def get_limb_direction(arm, closest_degrees=45):
  # should also use atan2 but I don't want to do more math
  dy = arm[2]['y'] - arm[0]['y'] # wrist -> shoulder
  dx = arm[2]['x'] - arm[0]['x']
  angle = degrees(atan(dy/dx))
  if (dx < 0):
    angle += 180

  # collapse to nearest closest_degrees; 45 for semaphore
  mod_close = angle % closest_degrees
  angle -= mod_close
  if mod_close > closest_degrees/2:
    angle += closest_degrees

  angle = int(angle)
  if angle == 270:
    angle = -90

  return angle

def is_arm_crossed(elbow, wrist, max_dist):
  return dist.euclidean([elbow['x'], elbow['y']], [wrist['x'], wrist['y']]) < max_dist

def is_arms_crossed(elbowL, wristL, elbowR, wristR, mouth_width):
  max_dist = mouth_width * ARM_CROSSED_RATIO
  return is_arm_crossed(elbowL, wristR, max_dist) and is_arm_crossed(elbowR, wristL, max_dist)

def is_leg_lifted(leg):
  if is_missing(leg):
    return False
  dy = leg[1]['y'] - leg[0]['y'] # knee -> hip
  dx = leg[1]['x'] - leg[0]['x']
  angle = degrees(atan2(dy, dx))
  return angle > LEG_LIFT_MIN

def is_jumping(hipL, hipR):
  global last_frames

  if is_missing([hipL, hipR]):
    return False

  last_frames[-1]['hipL_y'] = hipL['y']
  last_frames[-1]['hipR_y'] = hipR['y']

  if (hipL['y'] > last_frames[-2]['hipL_y'] + JUMP_THRESHOLD) and (
      hipR['y'] > last_frames[-2]['hipR_y'] + JUMP_THRESHOLD):
    last_frames[-1]['hips_dy'] = 1 # rising
  elif (hipL['y'] < last_frames[-2]['hipL_y'] - JUMP_THRESHOLD) and (
        hipR['y'] < last_frames[-2]['hipR_y'] - JUMP_THRESHOLD):
    last_frames[-1]['hips_dy'] = -1 # falling
  else:
    last_frames[-1]['hips_dy'] = 0 # not significant dy

  # consistently rising first half, lowering second half
  jump_up = all(frame['hips_dy'] == 1 for frame in last_frames[:HALF_HISTORY])
  get_down = all(frame['hips_dy'] == -1 for frame in last_frames[HALF_HISTORY:])

  return jump_up and get_down

def is_mouth_covered(mouth, palms):
  if is_missing(palms):
    return False
  dxL = (mouth[0]['x'] - palms[0]['x'])
  dyL = (mouth[0]['y'] - palms[0]['y'])
  dxR = (mouth[1]['x'] - palms[1]['x'])
  dyR = (mouth[1]['y'] - palms[1]['y'])
  return all(abs(d) < MOUTH_COVER_THRESHOLD for d in [dxL, dyL, dxR, dyR])

def is_squatting(hipL, kneeL, hipR, kneeR):
  if is_missing([hipL, kneeL, hipR, kneeR]):
    return False
  dyL = abs(hipL['y'] - kneeL['y'])
  dyR = abs(hipR['y'] - kneeR['y'])
  return (dyL < SQUAT_THRESHOLD) and (dyR < SQUAT_THRESHOLD)


def is_query_next(thumb, finger, min_finger_reach=20):
  d_finger = dist.euclidean([finger['x'], finger['y']], [thumb['x'], thumb['y']])
  print('distance = ', d_finger)
  return d_finger > min_finger_reach

def is_finger_out(finger, palmL, palmR, min_finger_reach):
  dL_finger = dist.euclidean([finger['x'], finger['y']], [palmL['x'], palmL['y']])
  dR_finger = dist.euclidean([finger['x'], finger['y']], [palmR['x'], palmR['y']])
  d_finger = min(dL_finger, dR_finger)
  return d_finger > min_finger_reach

def is_hand_open(thumb, forefinger, pinky, palmL, palmR, min_finger_reach):
  thumb_out = is_finger_out(thumb, palmL, palmR, min_finger_reach)
  forefinger_out = is_finger_out(forefinger, palmL, palmR, min_finger_reach)
  pinky_out = is_finger_out(pinky, palmL, palmR, min_finger_reach)
  return thumb_out and forefinger_out and pinky_out

def type_semaphore(armL_angle, armR_angle, image, shift_on, numerals, command_on, control_on,
  display_only, allow_repeat):
  global current_semaphore

  arm_match = SEMAPHORES.get((armL_angle, armR_angle), '')
  if arm_match:
    current_semaphore = arm_match.get('n', '') if numerals else arm_match.get('a', '')
    type_and_remember(image, shift_on, command_on, control_on, display_only, allow_repeat)
    return current_semaphore

  return False

def type_and_remember(image=None, shift_on=False, command_on=False, control_on=False,
  display_only=True, allow_repeat=False):
  global current_semaphore, last_keys

  if len(current_semaphore) == 0:
    return

  keys = []
  if shift_on:
    keys.append('shift')
  if command_on:
    keys.append('command')
  if control_on:
    keys.append('control')

  keys.append(current_semaphore)

  if allow_repeat or (keys != last_keys):
    last_keys = keys.copy()
    current_semaphore = ''
    output(keys, image, display_only)

def get_key_text(keys):
  if not (len(keys) > 0):
    return ''

  semaphore = keys[-1]
  keystring = ''
  if 'shift' in keys:
    keystring += 'S+'
  if 'command' in keys:
    keystring += 'CMD+'
  if 'control' in keys:
    keystring += 'CTL+'

  keystring += semaphore
  return keystring

def output(keys, image, display_only=True):
  keystring = '+'.join(keys)
  if len(keystring):
    print("keys:", keystring)
    if not display_only:
      keyboard.press_and_release(keystring)
    else:
      to_display = get_key_text(keys)
      cv2.putText(image, to_display, frame_midpoint,
        cv2.FONT_HERSHEY_SIMPLEX, 10, (0,0,255), 10)

def draw_star(image, center, radius):
  center_x ,center_y = center
  radius_outer, radius_inner = radius

  points = []
  for i in range(5):
    angle = i * 2 * np.pi / 5
    x_outer = center_x + int(radius_outer * np.cos(angle))
    y_outer = center_y + int(radius_outer * np.sin(angle))
    points.append((x_outer, y_outer))

    angle += np.pi / 5
    x_inner = center_x + int(radius_inner * np.cos(angle))
    y_inner = center_y + int(radius_inner * np.sin(angle))
    points.append((x_inner, y_inner))

  # 绘制五角星
  color = (0, 0, 255)  # 蓝色
  cv2.polylines(image, [np.array(points)], isClosed=True, color=color, thickness=5)


def render_and_maybe_exit(image, recording):
  cv2.imshow('Semaphore', image)
  if recording:
    recording.write(image)
  return cv2.waitKey(5) & 0xFF == 27


def main():
  global last_frames, frame_midpoint

  parser = argparse.ArgumentParser()
  parser.add_argument('--input', '-i', help='Input video device or file (number or path), defaults to 0', default='0')
  parser.add_argument('--flip', '-f', type=bool, default=False, help='Set to any value to flip resulting output (selfie view)')
  parser.add_argument('--landmarks', '-l', type=bool, default=True, help='Set to draw body landmarks')
  parser.add_argument('--record', '-r', type=bool, default=False, help='Set to save a timestamped AVI in current directory')
  parser.add_argument('--type', '-t', help='Set to any value to type output rather than only display')
  parser.add_argument('--repeat', '-p', help='Set to any value to allow instant semaphore repetitions')
  args = parser.parse_args()

  INPUT = int(args.input) if (args.input and args.input.isdigit()) else args.input
  FLIP = args.flip is not None
  DRAW_LANDMARKS = args.landmarks is not None
  RECORDING = args.record is not None
  DISPLAY_ONLY = args.type is None
  ALLOW_REPEAT = args.repeat is not None

  cap = cv2.VideoCapture(INPUT)

  frame_size = (int(cap.get(3)), int(cap.get(4)))
  frame_midpoint = (int(frame_size[0]/2), int(frame_size[1]/2))

  recording = cv2.VideoWriter(RECORDING_FILENAME,
    cv2.VideoWriter_fourcc(*'MJPG'), FPS, frame_size) if RECORDING else None

  emoji_img_list = []
  IMG_CNT = 8
  for idx in range(IMG_CNT):
      img = cv2.imread(str(idx)+'.jpeg', cv2.COLOR_BGR2RGB)[:,::-1,:]
      emoji_img_list.append(img)


  hands_on_off_cnt = 0
  query_flag = False
  color = [np.random.randint(255) for _ in range(3)]
  with mp.solutions.pose.Pose() as pose_model:
    with mp.solutions.hands.Hands(max_num_hands=2) as hands_model:
      while cap.isOpened():
        success, image = cap.read()
        if not success: break

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #pose_results = pose_model.process(image)
        hand_results = hands_model.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        hands = []
        hand_index = 0
        if hand_results.multi_hand_landmarks:
          for hand_landmarks in hand_results.multi_hand_landmarks:
            # draw hands
            if DRAW_LANDMARKS:
              mp.solutions.drawing_utils.draw_landmarks(
                image,
                hand_landmarks,
                mp.solutions.hands.HAND_CONNECTIONS,
                DEFAULT_LANDMARKS_STYLE,
                DEFAULT_HAND_CONNECTIONS_STYLE)
            hands.append([])
            for point in hand_landmarks.landmark:
              hands[hand_index].append({
                'x': point.x,
                'y': point.y
              })
            hand_index += 1
        if len(hands) > 0:
          hand = hands[0]
          Y, X, _ = image.shape
          thumb, forefinger = hand[4], hand[8]




          if not is_query_next(thumb, forefinger, 0.1):
            query_flag = False
            hands_on_off_cnt += 1
            color = [np.random.randint(255) for _ in range(3)]
          else:
            query_flag = True

          if is_thumb_red_heart(hand[3], hand[4], hand[6], hand[8]):
            query_flag = True
          else:
            query_flag = False
          if query_flag:
            thumb_pos = np.abs(np.array([thumb['x']*X, thumb['y']*Y])).astype(np.uint16)
            forefinger_pos = np.abs(np.array([forefinger['x']*X, forefinger['y']*Y])).astype(np.uint16)
            bottom_right = np.maximum(thumb_pos, forefinger_pos)
            top_left = np.minimum(thumb_pos, forefinger_pos)
            #cv2.rectangle(image, top_left, bottom_right, color, thickness=10)
            emoji_idx = hands_on_off_cnt % IMG_CNT
            emoji_idx = 7
            distance = (get_distance_from_ldmks(hand[8], hand[4], X=X, Y=Y)).astype(np.int16)
            mid_pt = (get_mid_point_from_ldmks(hand[8], hand[4], X=X, Y=Y)).astype(np.int16)
            mid_pt[1] -= 5
            xy_min, xy_max = mid_pt - distance//2, mid_pt + distance//2
            xy_min = np.maximum(xy_min, np.array([0, 0]))
            xy_max = np.minimum(xy_max, np.array([X, Y]))
            bbox_size = xy_max-xy_min
            if (bbox_size == 0).any():
              continue
            print('distance = ', distance)
            overlap_img = cv2.resize(emoji_img_list[emoji_idx], bbox_size)
            print(overlap_img.shape, xy_max-xy_min, image.shape, xy_max, xy_min)
            heart_xy_idx = np.where(overlap_img < 200)
            ROI_img = image[xy_min[1]:xy_max[1], xy_min[0]:xy_max[0]]
            ROI_img[heart_xy_idx] = overlap_img[heart_xy_idx]
            position = np.zeros([2]).astype(np.uint16)
            position[0] = max(xy_max[0], xy_min[0])
            position[1] = min(xy_max[1], xy_min[1])
            position = np.maximum(position, np.array([0, 0]))
            position = np.minimum(position, np.array([X, Y]))

            text = 'Morning! CanCan!'
            image = cv2.flip(image, flipCode=1)
            position[0] = X - position[0]
            cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, 1, color, thickness=2)
            image = cv2.flip(image,flipCode=1)
            '''
            center = (top_left + bottom_right)//2
            radius = 50
            #cv2.circle(image, center, radius, color, thickness=10)
            #draw_star(image, center, radius=[10, 50])

            print('Bug--> ', bottom_right - top_left)
            if (np.abs(bottom_right - top_left) == 0).any():
              continue
            overlap_img = cv2.resize(emoji_img_list[emoji_idx], np.abs(bottom_right - top_left))
            image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = overlap_img
            # alpha changing , no need now.
            #image = cv2.addWeighted(emoji_img_list[0], 1, image, 0.5, 0)
            '''


        if FLIP:
          image = cv2.flip(image, 1) # selfie view
        # draw pose
        if False:
        #if DRAW_LANDMARKS and False:
          mp.solutions.drawing_utils.draw_landmarks(
            image,
            pose_results.pose_landmarks,
            mp.solutions.pose.POSE_CONNECTIONS,
            DEFAULT_LANDMARKS_STYLE)
        #if pose_results.pose_landmarks and False:
        if False:
          # prepare to store most recent frame of movement updates over time
          last_frames = last_frames[1:] + [empty_frame.copy()]

          # short cool off period of last_frames for each sign
          if any(frame['signed'] for frame in last_frames):
            if render_and_maybe_exit(image, recording):
              break
            else:
              continue

          body = []
          # (0,0) bottom left to (1,1) top right
          for point in pose_results.pose_landmarks.landmark:
            body.append({
              'x': 1 - point.x,
              'y': 1 - point.y,
              'visibility': point.visibility
            })

          # cover mouth: backspace
          mouth = (body[9], body[10])
          palms = (body[19], body[20])
          if is_mouth_covered(mouth, palms):
            output(['backspace'], image, DISPLAY_ONLY)

          # command: left leg lift
          legL = (body[23], body[25], body[27]) # L hip, knee, ankle
          command_on = is_leg_lifted(legL)

          # control: right leg lift
          legR = (body[24], body[26], body[28]) # R hip, knee, ankle
          control_on = is_leg_lifted(legR)

          shoulderL, elbowL, wristL = body[11], body[13], body[15]
          armL = (shoulderL, elbowL, wristL)

          shoulderR, elbowR, wristR = body[12], body[14], body[16]
          armR = (shoulderR, elbowR, wristR)

          mouth_width = abs(mouth[1]['x']-mouth[0]['x'])

          # arrow keys: arms crossed + leg angles
          if is_arms_crossed(elbowL, wristL, elbowR, wristR, mouth_width):
            if is_limb_pointing(*legL) and is_limb_pointing(*legR):
              legL_angle = get_limb_direction(legL, LEG_ARROW_ANGLE)
              legR_angle = get_limb_direction(legR, LEG_ARROW_ANGLE)
              leg_arrow = leg_arrow_angles.get((legL_angle, legR_angle), '')
              if leg_arrow:
                output([leg_arrow + ' arrow'], image, DISPLAY_ONLY)

          # shift: both hands open
          shift_on = len(hands) > 0
          min_finger_reach = FINGER_MOUTH_RATIO * mouth_width
          palmL, palmR = body[17], body[18]
          for hand in hands:
            thumb, forefinger, pinky = hand[4], hand[8], hand[20]
            hand_open = is_hand_open(thumb, forefinger, pinky, palmL, palmR, min_finger_reach)
            shift_on = shift_on and hand_open

          # numbers: squat
          kneeL, kneeR = body[25], body[26]
          hipL, hipR = body[23], body[24]
          numerals = is_squatting(hipL, kneeL, hipR, kneeR)

          # alphanumeric: arm flags
          if is_limb_pointing(*armL) and is_limb_pointing(*armR):
            armL_angle = get_limb_direction(armL)
            armR_angle = get_limb_direction(armR)
            if type_semaphore(armL_angle, armR_angle, image,
              shift_on, numerals, command_on, control_on, DISPLAY_ONLY, ALLOW_REPEAT):
              last_frames[-1]['signed'] = True

          # repeat last: jump (hips rise + fall)
          # TODO: if ankles are always in view, could be more accurate than hips
          if is_jumping(hipL, hipR):
            output(last_keys, image, DISPLAY_ONLY)

        if render_and_maybe_exit(image, recording):
          break

  if RECORDING:
    recording.release()

  cap.release()
  cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
