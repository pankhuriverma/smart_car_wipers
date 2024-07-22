import cv2
import os




def createFrames(cam):

    
    # frame
    currentframe = 0
    # Initialize frame count and image file index
    frame_count = 0
    image_index = 1
        
    while(True):
            
            # reading frame
            ret,frame = cam.read()
        
            if ret == False:
                break
            
            # Save frame as image file
            if frame_count % frame_interval == 0:
                name = 'frame_and_mask_generation/frames/image' + str(image_index) + '.jpg'
                print ('Creating ...' + name)
        
                cv2.imwrite(name, frame)
                image_index += 1
            
            # Increment frame count
            frame_count += 1

    return image_index
       
   


if __name__== '__main__':
    
    #for index in range(1,48):
        # video capture
        cam = cv2.VideoCapture('frame_and_mask_generation/videos/47.mp4')
        # Get video frame rate and calculate frame interval
        print("using video frame_and_mask_generation/videos/47.mp4")

        fps = cam.get(cv2.CAP_PROP_FPS)
        frame_interval = int(30)
        
        index = createFrames(cam)
        print("index:  " + str(index))
        
    
        cam.release()
        cv2.destroyAllWindows()




    