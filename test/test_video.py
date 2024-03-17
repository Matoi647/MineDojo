import minedojo
import cv2
import numpy as np

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video = cv2.VideoWriter('survival_sword_food.mp4', fourcc, 20.0, (256, 160))

env = minedojo.make(
    task_id="survival_sword_food",
    # image_size=(160, 256)
    image_size=(640, 1024)
)
# env = minedojo.make(
#     task_id="harvest_wool_with_shears_and_sheep",
#     image_size=(160, 256)
# )
obs = env.reset()
for i in range(1000):
    act = env.action_space.no_op()
    act[0] = 1    # forward/backward
    if i % 10 == 0:
        act[2] = 1    # jump
        act[4] += 1
    if i % 8 == 0:
        act[4] -= 1
    obs, reward, done, info = env.step(act)

    print(i)
    # (channels, height, width) -> (height, width, channels)
    img = np.transpose(obs['rgb'], (1, 2, 0))
    output_video.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

output_video.release()
env.close()
