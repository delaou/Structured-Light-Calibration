import numpy as np
import cv2
import os
import matplotlib.pyplot as plt



class Calibrator(object): 
    
    def __init__(self): 
        
        self.world_coordinates_list = []
        self.camera_coordinates_list = []
        self.projector_coordinates_list = []
        self.fail_read = []
        self.count = 0
    
    def _read_sequence(self, root, name_sequence): 
        
        name_imgs = os.listdir(os.path.join(root, name_sequence))
        name_imgs.sort()
                
        self.imgs = []
        for index, name_img in enumerate(name_imgs): 
            dir_img = os.path.join(root, name_sequence, name_img)
            img = cv2.imread(dir_img, 0).astype(np.float32)
            self.imgs.append(img)
        print(f"{name_sequence}: {index + 1} images has been read")
        
        return dir_img
    
    
    def _calibrate_cam_center(self, img_size, proj_size, proj_offset, save_dir): 
        
        print("Start calibrating")  

        proj_size_mod = (proj_size[0], proj_size[1] * (1 + proj_offset))
        ret, camMat, camDist, camR, camT = cv2.calibrateCamera(self.world_coordinates_list, self.camera_coordinates_list, img_size, None, None)
        ret, projMat, projDist, projR, projT = cv2.calibrateCamera(self.world_coordinates_list, self.projector_coordinates_list, proj_size_mod, None, None)
        
        # ret, projMat, projDist, camMat, camDist, R, T, E, F = \
        #     cv2.stereoCalibrate(self.world_coordinates_list, self.projector_coordinates_list, self.camera_coordinates_list, projMat, projDist, camMat, camDist, img_size)
        
        ret, camMat, camDist, projMat, projDist, R, T, E, F = \
            cv2.stereoCalibrate(self.world_coordinates_list, self.camera_coordinates_list, self.projector_coordinates_list, camMat, camDist, projMat, projDist, img_size)

        np.savez(save_dir, camMat=camMat, camDist=camDist, projMat=projMat, projDist=projDist, R=R, T=T)
        
        print("Calibration completed\n")
        
        return camMat, camDist, projMat, projDist, R, T
    
    
    def _calibrate_proj_center(self, img_size, proj_size, proj_offset, save_dir): 
        
        print("Start calibrating")  
        
        proj_size_mod = (proj_size[0], proj_size[1] * (1 + proj_offset))
        ret, camMat, camDist, camR, camT = cv2.calibrateCamera(self.world_coordinates_list, self.camera_coordinates_list, img_size, None, None)
        ret, projMat, projDist, projR, projT = cv2.calibrateCamera(self.world_coordinates_list, self.projector_coordinates_list, proj_size, None, None)
        
        ret, projMat, projDist, camMat, camDist, R, T, E, F = \
            cv2.stereoCalibrate(self.world_coordinates_list, self.projector_coordinates_list, self.camera_coordinates_list, projMat, projDist, camMat, camDist, img_size)

        np.savez(save_dir, camMat=camMat, camDist=camDist, projMat=projMat, projDist=projDist, R=R, T=T)
        
        print("Calibration completed\n")
        
        return camMat, camDist, projMat, projDist, R, T
    
    
    def _extract_coordinates_2x2x3ps(self, proj_size, num_point, square_length, freq, dir_img): 
        
        print("Start extracting")
        
    #=================extract world coordinates=====================#
        world_coordinates = np.zeros((num_point[0] * num_point[1], 3), dtype=np.float32)
        x, y = np.mgrid[0:num_point[0], 0:num_point[1]]
        world_coordinates[:, :2] = square_length * cv2.merge((x.T, y.T)).reshape(-1, 2)
        
    #=================extract camera coordinates====================#       
        orig = np.round((self.imgs[0] + self.imgs[1] + self.imgs[2]) / 3).astype(np.uint8) 
        # plt.imshow(orig, cmap="gray")   
        # plt.show()
        # img_size = orig.shape[::-1]  
        ret, camera_coordinates = cv2.findChessboardCorners(orig, num_point, cv2.CALIB_CB_FAST_CHECK)
        camera_coordinates = cv2.cornerSubPix(orig, camera_coordinates, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        # print(camera_coordinates)
        if ret == False: 
            print("Fail to extract corners")
            self.fail_read.append(dir_img)
            return dir_img
        
    #=================extract projector coordinates=================#
        phase_vm_wrapped = cv2.phase(2 * self.imgs[0] - self.imgs[2] - self.imgs[1], np.sqrt(3.0) * (self.imgs[1] - self.imgs[2]), True)
        phase_vs = cv2.phase(2 * self.imgs[3] - self.imgs[5] - self.imgs[4], np.sqrt(3.0) * (self.imgs[4] - self.imgs[5]), True)
        
        # plt.imshow(phase_vs)
        # plt.show()
        
        steps = np.round((phase_vs * freq - phase_vm_wrapped) / 2 / np.pi)
        phase_vm = (steps * 2 * np.pi + phase_vm_wrapped) / freq
        # print(phase_vm.shape)
        
        phase_hm_wrapped = cv2.phase(2 * self.imgs[6] - self.imgs[8] - self.imgs[7], np.sqrt(3.0) * (self.imgs[7] - self.imgs[8]), True)
        phase_hs = cv2.phase(2 * self.imgs[9] - self.imgs[11] - self.imgs[10], np.sqrt(3.0) * (self.imgs[10] - self.imgs[11]), True)
        
        steps = np.round((phase_hs * freq - phase_hm_wrapped) / 2 / np.pi)
        phase_hm = (steps * 2 * np.pi + phase_hm_wrapped) / freq
        
        projector_coordinates = np.zeros_like(camera_coordinates)
        for index in range(len(camera_coordinates)): 
            proj_x = phase_vm[round(camera_coordinates[index][0][1]), round(camera_coordinates[index][0][0])] * proj_size[0] * 0.5 / np.pi
            proj_y = phase_hm[round(camera_coordinates[index][0][1]), round(camera_coordinates[index][0][0])] * proj_size[1] * 0.5 / np.pi
            projector_coordinates[index][0][0] = proj_x
            projector_coordinates[index][0][1] = proj_y  
        # print(projector_coordinates)
        print("Extract successfully\n")   
          
    #==================push coordinates=============================#
        sample = cv2.merge((orig, orig, orig))
        cv2.drawChessboardCorners(sample, num_point, camera_coordinates, ret)
        plt.imshow(sample)  
        plt.show()
        proj_is = np.zeros((*proj_size[::-1], 3), dtype=np.uint8)
        cv2.drawChessboardCorners(proj_is, num_point, projector_coordinates, ret)
        plt.imshow(proj_is)
        plt.show()
        self.world_coordinates_list.append(world_coordinates)
        self.camera_coordinates_list.append(camera_coordinates)
        self.projector_coordinates_list.append(projector_coordinates)
        self.count += 1
        
        return dir_img
    
    
    def _extract_coordinates_mngray(self, proj_size, num_point, square_length, threshold, m, dir_img): 
        
        print("Start extracting")
        
    #=================extract world coordinates=====================#
        world_coordinates = np.zeros((num_point[0] * num_point[1], 3), dtype=np.float32)
        x, y = np.mgrid[0:num_point[0], 0:num_point[1]]
        world_coordinates[:, :2] = square_length * cv2.merge((x.T, y.T)).reshape(-1, 2)
        
    #=================extract camera coordinates====================#       
        orig = self.imgs[1] 
        # plt.imshow(orig, cmap="gray")   
        # plt.show()
        # img_size = orig.shape[::-1]  
        ret, camera_coordinates = cv2.findChessboardCorners(orig, num_point, cv2.CALIB_CB_FAST_CHECK)
        camera_coordinates = cv2.cornerSubPix(orig, camera_coordinates, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        # print(camera_coordinates)
        if ret == False: 
            print("Fail to extract corners")
            self.fail_read.append(dir_img)
            return dir_img
        
    #=================extract projector coordinates=================#
        imgv_white = self.imgs[0]
        imgv_black = self.imgs[1]                                     
        imgsv_bin = []
        for index in range(m): 
            imgv_det = (self.imgs[index] - imgv_black) / (imgv_white - imgv_black)
            imgv_bin = np.zeros_like(imgv_white, dtype=bool)
            imgv_bin[imgv_det > threshold] = True
            imgsv_bin.append(imgv_bin)

        graycode_imgv = np.zeros((*(imgsv_bin[0].shape), len(imgsv_bin)))
        for index in range(m): 
            graycode_imgv[:, :, index] = self.imgs[index]
    
        bincode_imgv = np.zeros(graycode_imgv.shape, dtype=bool)
        bincode_imgv[:, :, 1] = graycode_imgv[:, :, 1]
        for index in range(2, graycode_imgv.shape[2]): 
            bincode_imgv[:, :, index] = np.logical_xor(bincode_imgv[:, :, index - 1], graycode_imgv[:, :, index])
            
        carry = np.array([2 ** (bincode_imgv.shape[2] - i - 1) for i in range(bincode_imgv.shape[2])])
        deccode_imgv = np.sum(bincode_imgv.astype(np.uint16) * carry, axis=2).astype(np.float32)
        
        
        imgh_white = self.imgs[m + 0]
        imgh_black = self.imgs[m + 1]                                     
        imgsh_bin = []
        for index in range(len(self.imgs) - m): 
            imgh_det = (self.imgs[m + index] - imgh_black) / (imgh_white - imgh_black)
            imgh_bin = np.zeros_like(imgh_white, dtype=bool)
            imgh_bin[imgh_det > threshold] = True
            imgsh_bin.append(imgh_bin)

        graycode_imgh = np.zeros((*(imgsh_bin[0].shape), len(imgsh_bin)))
        for index in range(len(self.imgs) - m): 
            graycode_imgh[:, :, index] = self.imgs[index]
    
        bincode_imgh = np.zeros(graycode_imgh.shape, dtype=bool)
        bincode_imgh[:, :, 1] = graycode_imgh[:, :, 1]
        for index in range(2, graycode_imgh.shape[2]): 
            bincode_imgh[:, :, index] = np.logical_xor(bincode_imgh[:, :, index - 1], graycode_imgh[:, :, index])
            
        carry = np.array([2 ** (bincode_imgh.shape[2] - i - 1) for i in range(bincode_imgh.shape[2])])
        deccode_imgh = np.sum(bincode_imgh.astype(np.uint16) * carry, axis=2).astype(np.float32)
        
        projector_coordinates = np.zeros_like(camera_coordinates)
        for index in range(len(camera_coordinates)): 
            proj_x = deccode_imgv[round(camera_coordinates[index][0][1]), round(camera_coordinates[index][0][0])] / np.max(deccode_imgv) * proj_size[0]
            proj_y = deccode_imgh[round(camera_coordinates[index][0][1]), round(camera_coordinates[index][0][0])] / np.max(deccode_imgh) * proj_size[1]
            projector_coordinates[index][0][0] = proj_x
            projector_coordinates[index][0][1] = proj_y  
        # print(projector_coordinates)
        print("Extract successfully\n")  
          
    #==================push coordinates=============================#
        # sample = cv2.merge((orig, orig, orig))
        # cv2.drawChessboardCorners(sample, num_point, camera_coordinates, ret)
        # plt.imshow(sample)  
        # plt.show()
        self.world_coordinates_list.append(world_coordinates)
        self.camera_coordinates_list.append(camera_coordinates)
        self.projector_coordinates_list.append(projector_coordinates)
        self.count += 1
        
        return dir_img
    
    
    def calibrate_2x2x3ps(self, num_point, root, square_length=6, freq=16, img_size=(2592, 1944), proj_size=(1280, 720), proj_offset=1, save_dir="data/calibrate_data", cam_center=True): 
        
        name_sequences = os.listdir(root)

        for name_sequence in name_sequences: 
            dir_img = self._read_sequence(root, name_sequence)     
            self._extract_coordinates_2x2x3ps(proj_size, num_point, square_length, freq, dir_img)
            
        if cam_center == True: 
            camMat, camDist, projMat, projDist, R, T = self._calibrate_cam_center(img_size, proj_size, proj_offset, save_dir)
            
        else: 
            camMat, camDist, projMat, projDist, R, T = self._calibrate_proj_center(img_size, proj_size, proj_offset, save_dir)
            
        return camMat, camDist, projMat, projDist, R, T



if __name__ == "__main__": 
    cal = Calibrator()
    camMat, camDist, projMat, projDist, R, T = cal.calibrate_2x2x3ps((11, 8), r"samples_folder/calibrate/camera0")
    print(f"camMat: {camMat}\n\ncamDist: {camDist}\n\nprojMat: {projMat}\n\nprojDist: {projDist}\n\nR: {R}\n\nT: {T}\n\n")
    