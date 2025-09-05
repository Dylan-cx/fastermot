import os
import os.path as osp
import cv2
import glob
import tqdm

 # generate video using OpenCV
def GenVideoFromImg(filename, output_video_path):
    '''
    :param frame_root: 存放图片打路径，这里可以根据需要自己修改
    :param output_video_path: 输出视频的名称，
    :return:
    '''
    data_root = os.path.join('/home/robot112/fastermot/src/lib/cfg/data', 'MOT16/train')
    frame_root = os.path.join(data_root, '..', 'outputs', filename)
    img_lists = glob.glob(frame_root + "/*.jpg")
    print("Image path is: ", frame_root)
    print("The length of img_lists: ", len(img_lists))
    img_lists.sort()
    img_test = cv2.imread(img_lists[0])
    h, w, c = img_test.shape
    fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
    # 为了仔细看效果，帧率调的很小
    video_writer = cv2.VideoWriter(output_video_path, fourcc, 10, (w, h))
    print("Begin to generate video!")
    for img_list in range(1,609):
    # for img_list in tqdm(img_lists):
        img = cv2.imread(img_list)
        video_writer.write(img)
    video_writer.release()

def ffm(filename):
    data_root = os.path.join('/home/robot112/fastermot/src/lib/cfg/data', 'MOT16/train')
    output_dir = os.path.join(data_root, '..', 'outputs', filename)
    output_video_path = osp.join(output_dir, '{}.mp4'.format("test"))
    cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -c:v copy {}'.format(output_dir, output_video_path)
    os.system(cmd_str)

if __name__ == '__main__':

    ffm('MOT17_test_realsense01')
    # GenVideoFromImg('MOT17_test_realsense01', 'testcv')

