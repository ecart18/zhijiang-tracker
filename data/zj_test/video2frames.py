import cv2
import os


videoroot = './level2_video'
all_videos = sorted([os.path.join(videoroot,i) for i in os.listdir(videoroot) if i.find('.mp4') != -1])
for video in all_videos:
  vidcap = cv2.VideoCapture(video)
  succ, image = vidcap.read()
  dirname = './level2' + video.split('.')[1] + '/img1'
  if not os.path.isdir(dirname):
    os.mkdir(dirname)
  count = 1
  succ = True
  while succ:
    cv2.imwrite(os.path.join(dirname,'{:0>6d}.jpg'.format(count)), image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])     # save frame as JPEG file
    succ, image = vidcap.read()
    print('save %6d in ' % count + dirname)
    count += 1
