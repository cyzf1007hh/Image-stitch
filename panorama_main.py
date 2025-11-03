from panorama import panorama
from option import Options
import cv2
import os
def main():
    opt = Options().parse()
    if not os.path.exists(opt.savepath):
        os.makedirs(opt.savepath)  # 如果文件夹不存在，创建文件夹
        print("拼接图文件夹已创建")
    else:
        print("拼接图文件夹已存在")

    if not os.path.exists(opt.savepointpath):
        os.makedirs(opt.savepointpath)  # 如果文件夹不存在，创建文件夹
        print("特征点文件夹已创建")
    else:
        print("特征点文件夹已存在")

    if not os.path.exists(opt.savematchpath):
        os.makedirs(opt.savematchpath)  # 如果文件夹不存在，创建文件夹
        print("图像配对文件夹已创建")
    else:
        print("图像配对文件夹已存在")
    PA = panorama(opt)

    PA.imageStitch_left()
    PA.imageStitch_right()
    print("done")

    method_name = opt.method
    print('method_name', method_name)
    cv2.imwrite(opt.savepath + method_name + "panorama_image.jpg", PA.result)
    print("image written")

    cv2.destroyAllWindows()

# python panorama_main.py --filepath datasets/picture/panorama/saoshedemo2 --savepath result/picture/panorama/saoshedemo2/
#python panorama_main.py --teaching True --filepath datasets/picture/panorama/teachingdemo3 --savepath result/picture/panorama/teachingdemo3/ --saveroot result/picture/panorama/teachingdemo3/ --save_keypoints True --save_match True --save_filter True

if __name__ == '__main__':
    main()