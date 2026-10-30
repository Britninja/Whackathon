from climbing_sim import ClimbingSim
import cv2

sim = ClimbingSim("N2")

image = sim.render("cam_left")

cv2.imwrite("perception/camera_left.png",cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

