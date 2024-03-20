import minedojo
import cv2
import numpy as np

height, width = 480, 768

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video = cv2.VideoWriter('survival_sword_food.mp4', fourcc, 20.0, (width, height))

env = minedojo.make(
    task_id="survival_sword_food",
    # image_size=(160, 256)
    image_size=(height, width)
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
        act[2] = 1      # jump
    if i % 5 == 0:
        if np.random.rand() > 0.5:
            act[4] += 1
        else:
            act[4] -= 1
    # if i % 10 == 2:
    #     act[4] += 1     # look right
    # if i % 10 == 6:     
    #     act[4] -= 1     # look left


    obs, reward, done, info = env.step(act)

    print(i)
    # (channels, height, width) -> (height, width, channels)
    img = np.transpose(obs['rgb'], (1, 2, 0))
    output_video.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

output_video.release()
env.close()
