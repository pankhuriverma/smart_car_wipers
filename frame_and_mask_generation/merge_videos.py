# Import everything needed to edit video clips
from moviepy.editor import *
 
# loading video dsa gfg intro video
clip1 = VideoFileClip("frame_and_mask_generation/parkingvideos/1.mp4")
clip2 = VideoFileClip("frame_and_mask_generation/parkingvideos/2.mp4")
 

 
# concatenating both the clips
final = concatenate_videoclips([clip1, clip2])
#writing the video into a file / saving the combined video
final.write_videofile("frame_and_mask_generation/parkingvideos/merged.mp4")