import numpy as np
import cv2 as cv
import numpy as np
from copy import deepcopy
import os
import sys

def find_contour(im_path):
    im = cv.imread(im_path)
    ori_img = deepcopy(im)
    assert im is not None, "file could not be read, check with os.path.exists()"
    imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(imgray, (1, 1), 0)
    ret, thresh = cv.threshold(blurred, 127, 255, 0)
    contours, heirarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    image = cv.drawContours(im, contours, -1, (0,255,0), 3)

    return contours, ori_img, blurred, thresh, image

def find_avg_area(contours):
    contours_area = [cv.contourArea(cont) for cont in contours]
    area_arr = np.array(contours_area)
    avg_area = np.average(area_arr, axis=0)
    print(f'average area: {avg_area}')
    return avg_area

def find_objects(contours, avg_area, im_name):
    total_objects = [cont for cont in contours if cv.contourArea(cont)> avg_area]
    print(f'total objects present in image {im_name} is : {len(total_objects)}')
    return total_objects

def show(ori_img, filtered_img, binary_img, contour_img, name):
    cv.imwrite(f"./ori/{name}", ori_img)
    cv.imwrite(f"./filtered/{name}", filtered_img)  
    cv.imwrite(f"./binary/{name}", binary_img)  
    cv.imshow(f"./contours/{name}", contour_img)   
    cv.waitKey(0)
    
      
def main(main_folder, visualize=False):
    im_paths = [os.path.join(main_folder, img_path) for img_path in os.listdir(main_folder) if img_path.endswith((".jpg", ".jpeg", ".png"))]
    for im_path in im_paths:
        im_name = im_path.split('/')[-1]
        contours, ori_img, filtered_img, binary_img, cont_img = find_contour(im_path=im_path)
        img = deepcopy(ori_img)
        avg_area = find_avg_area(contours=contours)
        # while True:
        total_objects = find_objects(contours=contours, avg_area=avg_area-40, im_name=im_name)
        only_objs = cv.drawContours(ori_img, total_objects, -1, (0,0,255), 2)
        image = deepcopy(img)
        cv.imshow("only objects", only_objs)
        i = 0
        for cont in total_objects:
            i+=1
            x,y,w,h = cv.boundingRect(cont)
            if x == 0 and y ==0:
                continue
            X = x + w
            Y = y + h
            x_center = x + w/2
            y_center = y + h/2
            image = cv.rectangle(image, (int(x), int(y)), (int(X), int(Y)), (0,255,0), 2)
            image = cv.putText(image, str(i), (int(x), int(y)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
            image = cv.circle(image, (int(x_center), int(y_center)), radius=2, color=(0,0,255), thickness=5)
        cv.imshow("bbox on objects", image)
        k = cv.waitKey(0)
        cv.destroyAllWindows()
        
        
        # if k == ord('m'):
        #     avg_area = avg_area+20
        #     print(f'new threshold area: {avg_area}')
        # elif  k == ord('n'):
        #     avg_area = avg_area-20
        #     print(f'new threshold area: {avg_area}')
        # elif k == ord('q'):
        #     break
        if visualize:
            show(ori_img, filtered_img, binary_img, cont_img, im_name)
        else:
            continue
        # cv.destroyAllWindows()
    
        
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f'Usage: python3 opencv_contour.py <folder containg images>')
    else:
        main_folder = sys.argv[1]
        main(main_folder=main_folder, visualize=False)
        