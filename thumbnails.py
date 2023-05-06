from openslide import open_slide
import matplotlib.pyplot as plt

slide_id="482937"
file_path="../mount_e/{}.svs".format(slide_id)
slide = open_slide(str(file_path))
img = slide.get_thumbnail((1000, 1000))
fig = plt.imshow(img)
plt.axis('off')
plt.savefig('../mount_outputs/weight_maps/{}_thumbnail.png'.format(slide_id), dpi=500,bbox_inches='tight')
plt.close()
