import numpy as np
from ImageStitch import imageStitcher
import cv2  
import os

class panorama(imageStitcher):
    """Stitch panoramic images for single machine strafing.
    
    Args:
        opt (class): Python class for resolving parameters.
    """
    def __init__(self, opt):
        super(panorama, self).__init__(opt)

        # Loading images and do some preparations.
        filenames = os.listdir(self.opt.filepath)
        for i in filenames:
            print('Loading...: ', self.opt.filepath + '/' + i)
        images = [cv2.imread(self.opt.filepath + '/' + filename) for filename in filenames]

        if self.opt.teaching == True:
            assert len(images) == 2

        sizes = np.zeros((len(images), 2))
        for index, img in enumerate(images):
            sizes[index,:] = img.shape[:2]
        max_size = np.max(sizes, axis = 0).astype('int')

        self.images = [cv2.resize(image, (max_size[1], max_size[0])) for image in images]

        self.count = len(self.images)
        self.left_list, self.right_list, self.center = [], [], None
        self.prepare_lists()

    def prepare_lists(self):
        """Divide the input image into left, center, and right intervals in order.

        """

        print('Number of images: ', self.count)
        self.centerIdx = self.count / 2 
        print('Center index image: ', self.centerIdx)

        self.center = self.images[int(self.centerIdx)]
        for i in range(self.count):
            if i <= self.centerIdx:
                self.left_list.append(self.images[i])
            else:
                self.right_list.append(self.images[i])
        print('Image lists prepared.')
        print('\n')   

    def imageStitch_left(self):
        """Stitch images in the left interval.

        """

        # Initiate the leftmost image.
        img1 = self.left_list[0]

        # Loop to add images to the right.
        for img2 in self.left_list[1:]:
            
            # Calculate the homography matrix by using methods from Python parent class 'imageStitcher'.
            kp1, des1, kp2, des2, _, _ = self.detectAndDescribe(img1, img2)
            good_matches, _ = self.matchKeypoints(img1, kp1, des1, img2, kp2, des2)
            H, _ = self.imageTransformation(img1, kp1, img2, kp2, good_matches)

            # Point: bottom - right. bottom right point is (col, row) while shape is row, col
            br = np.dot(H, np.array([img1.shape[1], img1.shape[0], 1]))
            # to guarantee h33 is 1
            br = br / br[-1]

            # Point: top - left.
            tl = np.dot(H, np.array([0, 0, 1]))
            tl = tl / tl[-1]

            # Point: bottom - left.
            bl = np.dot(H, np.array([0, img1.shape[0], 1]))
            bl = bl / bl[-1]

            # Point: top - right.
            tr = np.dot(H, np.array([img1.shape[1], 0, 1]))
            tr = tr / tr[-1]

            # Original a, add b, cause b stands for img2, b usually has a larger border in left - right.
            cx = int(max([0, img1.shape[1], tl[0], bl[0], tr[0], br[0]]))
            cy = int(max([0, img1.shape[0], tl[1], bl[1], tr[1], br[1]]))

            # To avoid negative coordinate.
            offset = [abs(int(min([0, img1.shape[1], tl[0], bl[0], tr[0], br[0]]))),
                      abs(int(min([0, img1.shape[0], tl[1], bl[1], tr[1], br[1]])))]
            
            # Image size for transformed image, large enough.
            dsize = (cx + offset[0], cy + offset[1])
            print("image dsize =>", dsize, "offset", offset)

            # Add offsets.
            tl[0:2] += offset; bl[0:2] += offset;  tr[0:2] += offset; br[0:2] += offset
            
            # Calculate the corner points' coordinate of img1 and img2.
            img1points = np.array([[0, 0], [0, img1.shape[0]], [img1.shape[1], 0], [img1.shape[1], img1.shape[0]]])
            img2points = np.array([tl, bl, tr, br])
            
            # Calculate the homography matrix after offset.
            H_off = cv2.findHomography(img1points, img2points)[0]
            # print('H_off', H_off)

            warped_img1 = cv2.warpPerspective(img1, H_off, dsize)
            warped_img2 = np.zeros([dsize[1], dsize[0], 3], np.uint8)
            warped_img2[offset[1]:img2.shape[0] + offset[1], offset[0]:img2.shape[1] + offset[0]] = img2
            
            #if self.opt.teaching == True:
            #    cv2.imwrite(self.opt.savepath + "warped_img1.jpg", warped_img1)
            #    cv2.imwrite(self.opt.savepath + "warped_img2.jpg", warped_img2)

            # Linear blending.
            temp = self.blend_linear(warped_img1, warped_img2)
            img1 = temp
        
        self.leftImage = temp

    def imageStitch_right(self):
        """Stitch images in the right interval.

        """

        # Loop to add images to the right.
        for img2 in self.right_list:
            img1 = self.leftImage

            # Calculate the homography matrix by using methods from Python parent class 'imageStitcher'.
            kp1, des1, kp2, des2, _, _ = self.detectAndDescribe(img1, img2)
            good_matches, _ = self.matchKeypoints(img1, kp1, des1, img2, kp2, des2)
            H, _ = self.imageTransformation(img1, kp1, img2, kp2, good_matches)

            # Inverse the homography matrix.
            H_T= np.linalg.inv(H)

            br = np.dot(H_T, np.array([img2.shape[1], img2.shape[0], 1]))
            br = br / br[-1]
            tl = np.dot(H_T, np.array([0, 0, 1]))
            tl = tl / tl[-1]
            bl = np.dot(H_T, np.array([0, img2.shape[0], 1]))
            bl = bl / bl[-1]
            tr = np.dot(H_T, np.array([img2.shape[1], 0, 1]))
            tr = tr / tr[-1]

            cx = int(max([0, img1.shape[1], tl[0], bl[0], tr[0], br[0]]))
            cy = int(max([0, img1.shape[0], tl[1], bl[1], tr[1], br[1]]))

            offset = [abs(int(min([0, img1.shape[1], tl[0], bl[0], tr[0], br[0]]))),
                      abs(int(min([0, img1.shape[0], tl[1], bl[1], tr[1], br[1]])))]
            
            dsize = (cx + offset[0], cy + offset[1])
            print("image dsize =>", dsize, "offset", offset)

            tl[0:2] += offset; bl[0:2] += offset; tr[0:2] += offset; br[0:2] += offset

            img1points = np.array([tl, bl, tr, br])
            img2points = np.array([[0, 0], [0, img2.shape[0]], [img2.shape[1], 0], [img2.shape[1], img2.shape[0]]])

            H_off = cv2.findHomography(img1points, img2points)[0]

            warped_img2 = cv2.warpPerspective(img2, H_off, dsize, flags = cv2.WARP_INVERSE_MAP)
            warped_img1 = np.zeros([dsize[1], dsize[0], 3], np.uint8)
            warped_img1[offset[1]:self.leftImage.shape[0] + offset[1], offset[0]:self.leftImage.shape[1] + offset[0]] = self.leftImage

            temp = self.blend_linear(warped_img1, warped_img2)
            self.leftImage = temp

        self.result = self.leftImage
    
    def blend_linear(self, warp_img1, warp_img2):
        """linear Blending(also known as Feathering), a method to eliminate image overlap.

        Args:
            warp_img1 (numpy.ndarray): The transfromed query image.
            warp_img2 (numpy.ndarray): The transfromed train image.
        """
        img1, img2 = warp_img1, warp_img2
        
        img1mask, img2mask = ((img1[:,:,0] | img1[:,:,1] | img1[:,:,2]) > 0), ((img2[:,:,0] | img2[:,:,1] | img2[:,:,2]) > 0)

        r1, c1 = np.nonzero(img1mask)
        out_1_center = [np.mean(r1), np.mean(c1)]
        
        r2, c2 = np.nonzero(img2mask)
        out_2_center = [np.mean(r2), np.mean(c2)]

        vec = np.array(out_2_center) - np.array(out_1_center)
        intsct_mask = img1mask & img2mask

        # Row and col index of nonzero element.
        r, c = np.nonzero(intsct_mask)

        out_wmask = np.zeros(img2mask.shape[:2])

        # Dot product of spherical coordinate and vec: measuring how much the vectors align or overlap in their directions.
        proj_val = (r - out_1_center[0]) * vec[0] + (c- out_1_center[1]) * vec[1]
        out_wmask[r,c] = (proj_val - (min(proj_val) + (1e-3))) / \
                        ((max(proj_val) - (1e-3)) - (min(proj_val) + (1e-3)))
        # blending
        mask1 = img1mask & (out_wmask==0)
        mask2 = out_wmask
        mask3 = img2mask & (out_wmask==0)
        
        out = np.zeros(img1.shape)
        for c in range(3):
            out[:,:,c] = img1[:,:,c] * (mask1 + (1 - mask2) * (mask2 != 0)) + \
                        img2[:,:,c] * (mask2 + mask3)

        return np.uint8(out)