# opencv_image_restoration
#图像恢复和彩色图片处理
        运行环境： ubuntu 14.04 opencv 3.0.0
        运行操作： cmake . 
                  make 
                  ./main
        
        加噪声(高斯噪声、椒盐噪声)//Add Gaussian noise and salt-and-pepper noise to input image
        
        均值滤波、统计排序滤波 //arithmetic mean filters, harmonic mean filters, contraharmonic mean filters, geometric
        mean filters, median filters, max filters and min filters
        
        彩色图片的处理 //Calculate the histogram on each channel separately, and then compute an average histogram
        from these three histograms.
