import numpy as np
import matplotlib.pyplot as plt
import cv2  
import time
import os

class imageStitcher:
    """Use the specified feature point extraction method to achieve the stitching of two images.
    
    Args:
        opt (class): Python class for resolving parameters.
    """
    def __init__(self, opt):
        self.opt = opt

        """
        Opt.Args:
            method (str): The method of image feature point extraction including.
                'SIFT', 'AKAZE', 'ORB', 'CenSurE' and 'FREAK'. Default: 'SIFT'.
        """
        method = self.opt.method
        # Initialize detector.
        if method == 'SIFT':
            self.detector = cv2.xfeatures2d.SIFT_create()
            # self.detector = cv2.SIFT_create()
        elif method == 'AKAZE':
            self.detector = cv2.AKAZE_create()
        elif method == 'ORB':
            self.detector = cv2.ORB_create()
        elif method == 'CenSurE':
            self.detector = cv2.xfeatures2d.StarDetector_create()
            self.computor = cv2.xfeatures2d.BriefDescriptorExtractor_create()
        elif method == 'BRISK':
            self.detector = cv2.BRISK_create()
        else:
            raise Exception('Parameter "method" error')

        self.method = method
        
    def image_show(self, showname, image, flags = cv2.WINDOW_KEEPRATIO, windows = False):
        """Use the function "imshow" in "cv2" to visualize image.
    
        Args:
            showname (str): Image name.
            image (numpy.ndarray): Image to be visualized.
            flags (int): Control the size of the window showed by "cv2".
        """
        if windows == True:
            cv2.namedWindow(showname, flags)
            cv2.imshow(showname, image)
            while True:
                if cv2.getWindowProperty(showname, 0) == -1: #当窗口关闭时为-1，显示时为0
                    break
                cv2.waitKey(1)
            # cv2.waitKey(0)
            cv2.destroyAllWindows()

    def hmerge_show(self, img1, img2, showname, flags = cv2.WINDOW_KEEPRATIO):
        """Horizontal stitching and visualization of two images.
    
        Args:
            img1 (numpy.ndarray): left image.
            img2 (numpy.ndarray): right image.
            showname (str): Stitched image name.
            flags (int): Control the size of the window showed by "cv2".
        """
        
        (h1, w1) = img1.shape[:2]
        (h2, w2) = img2.shape[:2]
        
        hmerge = np.zeros((max(h1, h2), w1 + w2, 3), dtype = "uint8")
        hmerge[0:h1, 0:w1] = img1
        hmerge[0:h2, w1: ] = img2
        
        self.image_show(showname, hmerge)
        
        return hmerge

    def detectAndDescribe(self, img1, img2):
        """Detection and visualization of image keypoints during registration process.
        
        Args:
            img1 (numpy.ndarray): Query image.
            img2 (numpy.ndarray): Train image.
        """
        
        # Convert the query and train image to grayscale.
        gray1, gray2 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        """
        Opt.Args:
            show_gray (bool): Control the visualization of grayscale query and train image. Default: False
        """
        # Horizontal stitching of grayscale images and display in the window.
        if self.opt.show_gray == True:
            self.hmerge_show(gray1, gray2, "gray")

        if self.method == 'CenSurE':
            # STAR feature detection.
            kpStar1, kpStar2 = self.detector.detect(img1, None), self.detector.detect(img2, None)

            # STAR feature description.
            kp1, des1 = self.computor.compute(img1, kpStar1)
            kp2, des2 = self.computor.compute(img2, kpStar2)
        else:
            # Use detectors to detect and calculate keypoints and descriptors directly.
            kp1, des1 = self.detector.detectAndCompute(img1, None)
            kp2, des2 = self.detector.detectAndCompute(img2, None)

        # Print out the number of keypoints detected in the query and train image.
        ## print("Key points in query image：" + str(len(kp1)))
        ## print("Key points in train image：" + str(len(kp2)))
    
        # Draw keypoints in the query and train image.
        # cv2.drawKeypoints(image, keypoints, outImage, color = None, flags = None)
        img3, img4 = cv2.drawKeypoints(img1, kp1, None), cv2.drawKeypoints(img2, kp2, None)

        # Draw the sizes and the orientations of the keypoints in the query and train image.
        img5 = cv2.drawKeypoints(img1, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        img6 = cv2.drawKeypoints(img2, kp2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        # Visualization of keypoints in image registration process.
        imgname1 = self.method + ' - keypoints in image registration process'
        hmerge1 = self.hmerge_show(img3, img4, imgname1)
        
        imgname2 = self.method + ' - sizes and orientations of keypoints in image registration process'
        hmerge2 = self.hmerge_show(img5, img6, imgname2)

        """
        Opt.Args:
            save_keypoints (bool): Control the saving of the query and train images with keypoints. Default: False
        """
        # Save the query and train images with keypoints.
        if self.opt.save_keypoints == True:
            #cv2.imwrite(self.opt.savepath + imgname1 + ".jpg", hmerge1)
            #cv2.destroyAllWindows()

            cv2.imwrite(self.opt.savepointpath + imgname2 + ".jpg", hmerge2)
            cv2.destroyAllWindows()

        return kp1, des1, kp2, des2, hmerge1, hmerge2

    def matchKeypoints(self, img1, kp1, des1, img2, kp2, des2):
        """Matching of image keypoints during image registration and visualization of the keypoints matching process.
        
        Args:
            img1 (numpy.ndarray): Query image.
            kp1 (tuple): Imformation of keypoints in query image.
            des1 (numpy.ndarray): Descriptors detected in query image.
            img2 (numpy.ndarray): Train image.
            kp2 (tuple): Imformation of keypoints in train image.
            des2 (numpy.ndarray): Descriptors detected in train image.
        """

        # Based on Brute Force matching descriptors, using BFMatcher objects to solve matching.
        if self.method == 'ORB' or self.method == 'CenSurE':
            bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        else:
            bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k = 2)
    
        ## print(matches[0]) -> Ex: (< cv2.DMatch 0000018621984C10>, < cv2.DMatch 0000018621B5A3D0>)
        ## print(len(matches))
    
        # Filter the correct paired keypoints.
        '''
        Dmatch structure: {
            queryIdx: 某一特征点在初始图像中的索引, 即img1中特征点的索引;
            trainIdx: 该特征点在另一张图像中相匹配特征点的索引, 即img2中特征点的索引;
            distance: 一对相匹配特征点描述符的欧式距离, 数值越小说明该对特征点越相近
        }
        '''
        good_matches = []
        for m, n in matches:
            """
            Opt.Args:
                ratio (float): Ratio used to retain paired keypoints. Default: 0.75
            """
            # m.distance: Closest distance; n.distance: Second closest distance.
            # When the ratio of the closest distance to the second closest distance is less than the ratio, retain the paired keypoints.
            if m.distance < self.opt.ratio * n.distance:
                good_matches.append([m])       

        # Determine the number of the retained paired keypoints.
        if len(good_matches) > 4:
            # Use KNN method to match keypoints and visualizating them in the window, flags = 2 for nearest neighbor matching.
            img7 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good_matches, None, flags = 2)

            # Visualization of keypoints matching in image registration process.
            self.image_show(self.method + " - BFmatch", img7)

            """
            Opt.Args:
                save_match (bool): Control the saving of the query and train images with matched keypoints. Default: False
            """
            # Save the query and train images after keypoints matching.
            if self.opt.save_match == True:
                imgname3 = self.method + ' - keypoints matching in image registration process'
                cv2.imwrite(self.opt.savematchpath + imgname3 + ".jpg", img7)
                cv2.destroyAllWindows()
        
            return good_matches, img7
        else:
            print('The number of good matched paired keypoints is not enough, you should expand the ratio used to retain paired keypoints or change method.')
            raise Exception('Keypoints not good match error!')

    def imageTransformation(self, img1, kp1, img2, kp2, good_matches):
        """Solving the homography matrix by using RANSAC algorithm and visualization of keypoints during image stitching process.
        
        Args:
            img1 (numpy.ndarray): Query image.
            kp1 (tuple): Imformation of keypoints in query image.
            img2 (numpy.ndarray): Train image.
            kp2 (tuple): Imformation of keypoints in train image.
            good_matches (list): A list obtained the information of retained paired keypoints.
        """

        kp1_np = np.float32([kp.pt for kp in kp1])
        kp2_np = np.float32([kp.pt for kp in kp2])

        # Store the index of two keypoints in the reserved matching pairs in the query and train images.
        keypoint_indexs = [(good_match[0].queryIdx, good_match[0].trainIdx) for good_match in good_matches]

        # Obtain the keypoint coordinates of matching pairs and consturct the two set of points.
        if self.method == 'ORB' or self.method == 'CenSurE':
            pts1 = np.float32([kp1[m[0].queryIdx].pt for m in good_matches]).reshape(-1,1,2)
            pts2 = np.float32([kp2[m[0].trainIdx].pt for m in good_matches]).reshape(-1,1,2)
        else:
            pts1 = np.float32([kp1_np[i] for (i, _) in keypoint_indexs])
            pts2 = np.float32([kp2_np[i] for (_, i) in keypoint_indexs])

        """
        Opt.Args:
            threshold (float): Maximum allowable reprojection error threshold for treating point pairs as inliers.
        """
        # Compute the homography matrix between the two sets of points.
        (H, status) = cv2.findHomography(pts1, pts2, cv2.RANSAC, self.opt.threshold)

        # Obtain the width and height of the query and train images.
        (h1, w1) = img1.shape[:2]
        (h2, w2) = img2.shape[:2]

        # Horizontal image stitching.
        hmerge3 = np.zeros((max(h1, h2), w1 + w2, 3), dtype = "uint8")
        hmerge3[0:h1, 0:w1] = img1
        hmerge3[0:h2, w1: ] = img2

        pt1 = []
        for ((queryIdx, trainIdx), s) in zip(keypoint_indexs, status):
            # When the paired points matching is successful, draw them onto the visualization image.
            if s == 1:
                pt1_ = (int(kp1_np[queryIdx][0]), int(kp1_np[queryIdx][1]))
                pt2_ = (int(kp2_np[trainIdx][0]) + w1, int(kp2_np[trainIdx][1]))
                pt1.append(pt1_)
                cv2.circle(hmerge3, pt1_, 1, (0, 0, 255), 2)
                cv2.circle(hmerge3, pt2_, 1, (0, 0, 255), 2)
                cv2.line(hmerge3, pt1_, pt2_, (0, 255, 255), 1)

        # Visualization of keypoints during image stitching process(After using RANSAC algorithm).
        self.image_show(self.method + " - FindHomography (RANSAC algorithm)", hmerge3)

        """
        Opt.Args:
            save_filter (bool): Control the saving of the query and train images with matched keypoints. Default: False
        """
        # Save the image after filtering keypoints through the homography matrix operation.
        #if self.opt.save_filter == True:
        #    imgname4 = self.method + ' - keypoints matching visualization with findHomography (RANSAC algorithm)'
        #    cv2.imwrite(self.opt.savepath + imgname4 + ".jpg", hmerge3)
        #    cv2.destroyAllWindows()
    
        return H, pt1

    def imageStitch(self, img1, img2, H, pt1 = None):
        """Image stitching and visualization of the final stitching result(with keypoints or not).
        
        Args:
            img1 (numpy.ndarray): Query image.
            img2 (numpy.ndarray): Train image.
            H (numpy.ndarray): The homography matrix.
            pt1 (list): Coordinates of keypoints in query image.
        """

        # Transform the train image to a different perspective by using the homography matrix and stitch it together in query image.
        img2_per = cv2.warpPerspective(img2, np.linalg.inv(H), (img1.shape[1] + img2.shape[1], max(img1.shape[0], img2.shape[0])))
        result = img2_per.copy()
        result[0:img1.shape[0], 0:img1.shape[1]] = img1
        
        # Visualize image stitching results using homography matrix directly.
        imgname5 = self.method + " - Image stitching result(using homography matrix directly)"
        self.image_show(imgname5, result)

        """
        Opt.Args:
            direct_result (bool): Control the saving of the stitching image with keypoints. Default: True
        """
        # Save image stitching result using homography matrix directly.
        if self.opt.direct_result == True:
            cv2.imwrite(self.opt.savepath + imgname5 + ".jpg", result)
            cv2.destroyAllWindows()

        result_clean = result.copy()

        # Draw the keypoints onto the image stitching result.
        if pt1 != None:
            for pt1_ in pt1:
                cv2.circle(result, pt1_, 1, (0, 0, 255), 2)

        # Visualize image stitching result with keypoints using homography matrix directly.
        imgname6 = self.method + " - Image stitching result with keypoints(using homography matrix directly)"
        self.image_show(imgname6, result)

        # Save image stitching result with keypoints using homography matrix directly.
        if self.opt.direct_result == True:
            cv2.imwrite(self.opt.savepath + imgname6 + ".jpg", result)
            cv2.destroyAllWindows()
    
        return img2_per, result_clean, result

    def imageBlender(self, img1, img2_per, pt1 = None):
        """Perform some weighting on the overlapping parts of the image to achieve smooth transitions of pixel values.
    
        Args:
            img1 (numpy.ndarray): Query image.
            img2_per (numpy.ndarray): The transfromed train image.
            pt1 (list): Coordinates of keypoints in query image.
        """

        # Use linear Blending(Feathering) method to eliminate image overlap.
        result_final_clean = self.linearBlending(img1, img2_per)

        # Remove the black border.
        result_final_clean = self.removeBlackBorder(result_final_clean)

        # Visualize image stitching result after blending.
        imgname7 = self.method + " - Image stitching result after Blending"
        self.image_show(imgname7, result_final_clean)

        """
        Opt.Args:
            blending_result (bool): Control the saving of the stitching image with keypoints. Default: True
        """
        # Save image stitching result after blending.
        if self.opt.blending_result == True:
            cv2.imwrite(self.opt.savepath + imgname7 + ".jpg", result_final_clean)
            cv2.destroyAllWindows()

        result_final = result_final_clean.copy()
        
        # Draw the keypoints onto the image stitching result.
        if pt1 != None:
            for pt1_ in pt1:
                cv2.circle(result_final, pt1_, 1, (0, 0, 255), 2)

        # Visualize image stitching result with keypoints after blending.
        imgname8 = self.method + " - Image stitching result with keypoints after Blending"
        self.image_show(imgname8, result_final)

        # Save image stitching result with keypoints after blending.
        if self.opt.blending_result == True:
            cv2.imwrite(self.opt.savepath + imgname8 + ".jpg", result_final)
            cv2.destroyAllWindows()
    
        return result_final_clean, result_final

    def linearBlending(self, img1, img2_per):
        """linear Blending(also known as Feathering), a method to eliminate image overlap.
    
        Args:
            img1 (numpy.ndarray): Query image.
            img2_per (numpy.ndarray): The transfromed train image.
            ## save (bool): Control the saving of the overlap mask image. Default: False
        """

        (h1, w1) = img1.shape[:2]
        (h2, w2) = img2_per.shape[:2]
        img1_mask = np.zeros((h2, w2), dtype = "uint8")
        img2_per_mask = np.zeros((h2, w2), dtype = "uint8")

        # Find the query image mask region(Those not zero pixels).
        for i in range(h1):
            for j in range(w1):
                if np.count_nonzero(img1[i, j]) > 0:
                    img1_mask[i, j] = 1

        # Find the transfromed train image mask region(Those not zero pixels).
        for i in range(h2):
            for j in range(w2):
                if np.count_nonzero(img2_per[i, j]) > 0:
                    img2_per_mask[i, j] = 1

        # Find the overlap mask(overlap region of query image and transfromed train image).
        overlap_mask = np.zeros((h2, w2), dtype = "uint8")
        for i in range(h2):
            for j in range(w2):
                if (np.count_nonzero(img1_mask[i, j]) > 0 and np.count_nonzero(img2_per_mask[i, j]) > 0):
                    overlap_mask[i, j] = 1

        # Plot the overlap mask
        plt.figure(21)
        plt.title(self.method + " - overlap_mask")
        plt.imshow(overlap_mask.astype(int), cmap = 'gray')
        plt.savefig(self.opt.savepath + self.method + ' - overlap_mask.jpg')

        # # Visualize the overlap mask image.
        # imgname9 = self.method + " - overlap mask"
        # self.image_show(imgname9, overlap_mask)

        # # Save the overlap mask image.
        # cv2.imwrite(self.opt.savepath + imgname9 + ".jpg", overlap_mask)
        # cv2.destroyAllWindows()
    
        # Compute the alpha mask to linear blending the overlap region.
        # Alpha value depends on query image.
        alpha_mask = np.zeros((h2, w2))
        for i in range(h2): 
            minIdx = maxIdx = -1
            for j in range(w2):
                if (overlap_mask[i, j] == 1 and minIdx == -1):
                    minIdx = j
                if (overlap_mask[i, j] == 1):
                    maxIdx = j

            # Represent this row's pixels are all zero, or only one pixel not zero
            if (minIdx == maxIdx):
                continue
                
            decrease_step = 1 / (maxIdx - minIdx)
            for j in range(minIdx, maxIdx + 1):
                alpha_mask[i, j] = 1 - (decrease_step * (j - minIdx))

        img_lB = np.copy(img2_per)
        img_lB[:h1, :w1] = np.copy(img1)
        
        # Linear blending process.
        for i in range(h2):
            for j in range(w2):
                if (np.count_nonzero(overlap_mask[i, j]) > 0):
                    img_lB[i, j] = alpha_mask[i, j] * img1[i, j] + (1 - alpha_mask[i, j]) * img2_per[i, j]
        
        return img_lB

    def removeBlackBorder(self, img):
        '''Remove the black border of the stitched image using homography matrix directly.

        Args:
            img1 (numpy.ndarray): Query image.
            img2 (numpy.ndarray): Train image.
            H (numpy.ndarray): The homography matrix.
            save (bool): Control the saving of the stitching image with keypoints. Default: True
        '''
        
        h, w = img.shape[:2]
        reduced_h, reduced_w = h, w
        
        # right to left.
        for j in range(w - 1, -1, -1):
            all_black = True
            for i in range(h):
                if np.count_nonzero(img[i, j]) > 0:
                    all_black = False
                    break
            if all_black == True:
                reduced_w = reduced_w - 1
                
        # bottom to top.
        for i in range(h - 1, -1, -1):
            all_black = True
            for j in range(reduced_w):
                if np.count_nonzero(img[i, j]) > 0:
                    all_black = False
                    break
            if all_black == True:
                reduced_h = reduced_h - 1
        
        return img[:reduced_h, :reduced_w]