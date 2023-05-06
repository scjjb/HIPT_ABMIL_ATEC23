import os
vipsbin = r'i:\vips\vips-dev-8.14\bin'
os.environ['PATH'] = vipsbin + ';' + os.environ['PATH']

import pyvips

tiffload("I:/treatment_data/2-1613704B.svs")
tiffsave("I:/treatment_data/pyramid/2-1613704B.svs",compression=jpeg,pyramid=True,tile=True)

