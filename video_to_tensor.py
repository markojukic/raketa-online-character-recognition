from PIL import Image, ImageDraw, ImageFilter, ImageOps
import numpy as np

def video_to_tensor(video, line_width=10, base_resolution=300, real_resolution=300):   
    """pretvara video(listu drawings-a) u numpy.array gdje je svaki slice slika jednog drawing-a"""
    
    #line_width-sirina linije kojom se znamenka crta
    #base_resolution-slika rezolucije base_resolution*base_resolution u kojoj se crta
    #real_resolution-rezolucija na koju se slika skalira na kraju
 
    first=True
    for drawing in video:
        scaled_strokes=[stroke*base_resolution for stroke in drawing.strokes]
        img = Image.new("RGB", (base_resolution, base_resolution))
        img1 = ImageDraw.Draw(img)
        for stroke in scaled_strokes:
            x=stroke[:,0]
            y=stroke[:,1]
            y=base_resolution-y-1
            img1.line(np.column_stack((x,y)), width = line_width) 
            
        gray_image = ImageOps.grayscale(img)
        gray_image=gray_image.filter(ImageFilter.GaussianBlur(radius = 3))
        gray_image=gray_image.resize((real_resolution,real_resolution))
        frontal_slice=np.asarray(gray_image)
        
        if first:
            T=frontal_slice[None]
            first=False
        else:
            T=np.concatenate((T,frontal_slice[None]),axis=0)
    return T
