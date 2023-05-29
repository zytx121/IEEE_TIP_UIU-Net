# UIU-Net: U-Net in U-Net for Infrared Small Object Detection

[Xin Wu](https://scholar.google.com/citations?user=XzV9xYIAAAAJ&hl=en), [Danfeng Hong](https://sites.google.com/view/danfeng-hong), [Jocelyn Chanussot](http://jocelyn-chanussot.net/)
---------------------

The code in this toolbox implements the ["UIU-Net: U-Net in U-Net for Infrared Small Object Detection"](https://ieeexplore.ieee.org/document/9989433).
More specifically, it is detailed as follow.

![alt text](./outline.jpg)




```bash
# 0. download HRSID dataset from https://github.com/open-mmlab/mmrotate/tree/main/tools/data/hrsid

# 1. convert json to binary mask
python json2mask.py

# 2. train
python train_sar_ship.py

# 3. generate binary mask
python test_sar_ship.py

# 4. Evaluate the rotated box mAP
python eval_sar_ship.py

```


We speculate that there are two main reasons why UIU-Net performs poorly in SAR ship detection tasks: 

- UIU-Net is unable to handle images with a large number of targets, and only focuses on the more prominent ones. There is a GAP between the optimization objective of the semantic segmentation task and the indicators of the SAR ship detection task.

- The binary mask output of the UIU-Net network cannot handle densely arranged targets well. When a binary mask is converted into a rotated box, multiple densely arranged targets will merge into one target, resulting in the miss of the target. 



Citation
---------------------

**Please kindly cite the papers if this code is useful and helpful for your research.**

X. Wu, D. Hong, J. Chanussot. UIU-Net: U-Net in U-Net for Infrared Small Object Detection, IEEE Trans. Image. Process., 2023, 32: 364-376. 

     @article{wu2023uiu,
      title     = {UIU-Net: U-Net in U-Net for Infrared Small Object Detection},
      author    = {X. Wu and D. Hong and J. Chanussot},
      journal   = {IEEE Trans. Image. Process.}, 
      volume    = {32},
      pages     = {364--376},
      year      = {2023},
      publisher = {IEEE}
     }


System-specific notes
---------------------
Please refer to the file of `requirements.txt` for the running enviroment of this code.

:exclamation: The model in `saved_models/uiunet` can be downloaded from the following baiduyun:

Baiduyun: https://pan.baidu.com/s/11JSqFKxq7XTvTzgOIWfdOg  (access code: eu9f)

Google drive: https://drive.google.com/file/d/1wZ_W5-Wj3wOt7kMB6nwa0xntZTD14sPZ/view?usp=share_link

Licensing
---------

Copyright (C) 2022 Danfeng Hong

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, version 3 of the License.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program.

Contact Information:
--------------------

Danfeng Hong: hongdanfeng1989@gmail.com<br>
Danfeng Hong is with the Aerospace Information Research Institute, Chinese Academy of Sciences, 100094 Beijing, China.

