import random

def corrupt_image(image,percent):
    img = image.copy()
    num,height,width,channels = img.shape
    for i in range(num):
        # only used for grey image
        if channels==1:
            for j in range(height):
                tmp_row=[k for k in range(width)]
                random.shuffle(tmp_row)
                tmp_row = tmp_row[0:int(width*percent)]
                for l in tmp_row:
                    img[i][j][l][0]=0
        
        if channels==3:
            for j in range(height):
                for channel in range(channels):
                    tmp_row=[k for k in range(width)]
                    random.shuffle(tmp_row)
                    tmp_row = tmp_row[0:int(width*percent)]
                    for l in tmp_row:
                        img[i][j][l][channel]=0
    
    return img

def corrupt_single_image(image,percent):
    img = image.copy()
    height,width = img.shape
    
    # only used for grey image

    for j in range(height):
        tmp_row=[k for k in range(width)]
        random.shuffle(tmp_row)
        tmp_row = tmp_row[0:int(width*percent)]
        for l in tmp_row:
            img[j][l]=0
    
   
    return img

                
        