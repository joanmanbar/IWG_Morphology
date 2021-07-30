## Introduction----
## Demo script for intermediate wheatgrass stem counting
## This script will find each of these features:
# 1) stems
## This script will quantify this for each feature
# 1) number of stems
# 2) diameter of stems
## Date: 03.29.2021
## Author: Garett Heineck
##############
###
###
###
###



###
###
###
###
##############
## Loading required packages and functions----
## Remember to UPDATE R!
R.Version()[c('version.string','nickname')]
## Hey! Is your version 4.0.4 "Lost Library"
## PLEASE UPDATE R (if you have not done so recently)
#************************#
## Required packages - some may need installation first
require(tidyverse)
require(readxl)
require(dplyr)
require(jpeg)
require(randomForest)
require(stringr)
require(ggplot2)
require(cowplot)
require(emmeans)
require(EBImage) #this needs to be downloaded from (https://www.bioconductor.org/packages/release/bioc/html/EBImage.html)***
require(plotly)
# OR use these lines
#if (!requireNamespace("BiocManager", quietly = TRUE))
#  install.packages("BiocManager")
#BiocManager::install("EBImage")
#************************#
ImgOverlay<- function(original.image, mask) #function for overlaying an RGB image onto a binary mask***
{
  replace<-which(mask==0)
  red<-original.image[,,1] 
  blue<-original.image[,,2]
  green<-original.image[,,3]
  red[replace]<-0 
  blue[replace]<-0
  green[replace]<-0
  return(array(c(red, blue, green), dim=dim(original.image)))
}
#************************#
ImgToDataframe<- function(image.path) #function for vecotizing an RGB image and maintaining its matrix coordinates***
{
  image.dat<- readImage(image.path)
  coor<- as.data.frame.table(image.dat[,,1])[1:2]
  red<- 255*as.data.frame.table(image.dat[,,1])[3] #to obtain an RGB color space multiply by 255***
  green<- 255*as.data.frame.table(image.dat[,,2])[3]
  blue<- 255*as.data.frame.table(image.dat[,,3])[3]
  image.dat<- bind_cols(coor, red, green, blue)
  colnames(image.dat)<- c("y","x","red","green","blue")
  return(image.dat)
}
##############
###
###
###
###



###
###
###
###
##############
## Enter file paths----
## Parent directory folder named "img_IWG_stem.demo.3.29.21"***
## To run this you need a folder named "IWG_img_output"***
img_IWG_stem.demo.3.29.21<- "/Volumes/Samsung USB" #NOTE: change to your own file path***
#************************#
## Creating folders to store all image output
#NOTE: the folder "original_img" is where the images you want to process need to be***
folders <- c("training data", 
             "results", 
             "original_img", 
             "crop_img",
             "glove_01",
             "glove_02",
             "stems_01",
             "stems_02",
             "stems_03"
             ) #adding in the correct folder list*** 
for (i in 1:length(folders))  { 
  dir.create(paste(img_IWG_stem.demo.3.29.21,folders[i], sep="/")) 
}
#NOTE: you may see Warning messages, that is ok as default setting will not overwrite existing folders***
#NOTE: make sure the "original_img" folder has images in it***
##############
###
###
###
###



###
###
###
###
##############
## Read in datasheet----
## The data contains information about each image captured in the original images folder
## This may be observations of the seeds taken in the field or lab or pedigree information
## You can open the excel spreadsheet to read the column descriptions
#************************#
IWG_seed.dat<- read_excel(paste(img_IWG_stem.demo.3.29.21, "results", "stem_dat.xlsx", sep = "/"), 
                              na = "NA") 
summary(IWG_seed.dat)
##############
###
###
###
###



###
###
###
###
##############
## Reducing orginal image pixel number----
#************************#
original_img.path<- list.files(path=paste(img_IWG_stem.demo.3.29.21, "original_img",sep = "/"), full.names = T)
original_img.name<- list.files(path=paste(img_IWG_stem.demo.3.29.21, "original_img",sep = "/"), full.names = F)
folder_crop_img<-   paste(img_IWG_stem.demo.3.29.21,"crop_img",sep = "/")
## This portions will reduce the number of pixels of the image through a resizing function
for(i in 1:length(original_img.path)){
  temp1<- readImage(original_img.path[i])
  temp2<- EBImage::resize(temp1, 
                          2000) #select your pixel width here***
  writeImage(temp2, paste(folder_crop_img, "/", original_img.name[i], sep = ""), quality = 100)
}
##############
###
###
###
###



###
###
###
###
##############
## Load the training data----
## Information on how to create training data can be found in the TRAINING DATA HELP GUIDE
## SEE: https://github.com/GarettHeineck/making-training-data
## Collectively the training mixes are called a palette and are saved in the training palette folder
## The palette has many RGB mixes, each helps in predicting different features within the image
#************************#
palette_directory_IWG.stem<- paste(img_IWG_stem.demo.3.29.21, "training data",sep = "/") #file path where mixes are saved***
#************************#
mixes_names<- list.files(path=palette_directory_IWG.stem,pattern="*.csv",full.names = FALSE) #name directory for what is in the palette folder***
mixes_path<- list.files(path=palette_directory_IWG.stem, pattern="*.csv", full.names = TRUE) #path directory for what is in the palette folder***
training.palette_IWG.stem<- data.frame()
#this for() loop will systematically rearrange and condense each mix file in the training palette folder***
#the reason I am doing this is to allow the script to update itself upon adding additional mixes***
for (i in 1:length(mixes_path)){
  temp_mix<- read.csv(mixes_path[i])
  temp_mix$band<- NA
  temp_mix$band[1:which(temp_mix$Label == "Red")] <- "Red"
  temp_mix$band[(which(temp_mix$Label == "Red")+1):which(temp_mix$Label == "Green")] <- "Green"
  temp_mix$band[(which(temp_mix$Label == "Green")+1):which(temp_mix$Label == "Blue")] <- "Blue"
  temp<- split(temp_mix, temp_mix$band)
  temp2<- do.call("cbind", split(temp_mix, temp_mix$band))
  image<- temp2$Blue.Label[i]
  mix<- mixes_names[i]
  temp3<- data.frame(mix, image, x=temp2[5]$Blue.X, y=temp2[6]$Blue.Y, red=temp2[18]$Red.Mean, green=temp2[11]$Green.Mean, blue=temp2[4]$Blue.Mean)
  training.palette_IWG.stem<- rbind(training.palette_IWG.stem, temp3) 
}
summary(training.palette_IWG.stem) #summarizing the training palette***
count(training.palette_IWG.stem, mix) #counting observations in each mix of the training palette*** 
##############
###
###
###
###



###
###
###
###
##############
## Random forest prediction models----
## Here were are detecting four features: 
# 1) glove
# 2) exclude glove and background from foreground (bundle of stems)
# 3) isolate and quantify count stems
## Initially, all four features will be discriminated simultaneously using a categorical response (as.factor())
## Further predictions will be made using a binomial response for each feature individually
#************************#
## Making a training palette is the first step in making a successful model
## Palettes are made up of mixes (a mix is a file containing training data)
## Each palette will probably be unique to a model with mixes added or subtracted as needed
## One must think about what features are being extracted using the model
palette_selection_categorical<- filter(training.palette_IWG.stem, !grepl("background", mix) & !grepl("circle", mix)) #creating an object with training data, here filtering can be done for different training mixes***
 
palette_selection_binary<- filter(training.palette_IWG.stem, !grepl("background", mix) & !grepl("circle", mix)) 

palette_selection_circle<- filter(training.palette_IWG.stem, !grepl("null", mix) & !grepl("pith", mix)  & !grepl("cortex", mix))
#************************#
palette_selection_categorical<- palette_selection_categorical %>%
  mutate(classification = case_when(grepl("null", mix) ~ 0, #re coding each mix type (null ergot...) with a numeric integer to help make the model***
                                    grepl("glove", mix) ~ 1,
                                    grepl("cortex", mix) ~ 2,
                                    grepl("pith", mix) ~ 1)) 
palette_selection_categorical %>% count(classification)
#palette_selection_categorical %>% group_by(mix) %>% summarise(avg=mean(classification)) %>% View() #viewing the classisifications for this training palette***
rfm_IWG_categorical<- randomForest(factor(classification)~(red+green+blue), #NOTE the factor() before the response***
                                data=palette_selection_categorical, 
                                ntree=50, #it is best to keep tres under 100***
                                mtry = 1, #with so few predictors 1 sample per tree is best***
                                importance=TRUE)
print(rfm_IWG_categorical)
plot(rfm_IWG_categorical)
importance(rfm_IWG_categorical) 
#************************#
palette_selection_stem<- palette_selection_binary %>%
  mutate(classification = case_when(grepl("null", mix) ~ 0, #re coding each mix type (null ergot...) with a numeric integer to help make the model***
                                    grepl("glove", mix) ~ 0,
                                    grepl("cortex", mix) ~ 1,
                                    grepl("pith", mix) ~ 1)) 

stem.rfm<- randomForest(classification~(red+green+blue),
                         data=palette_selection_stem, 
                         ntree=40,
                         mtry = 1,
                         importance=TRUE)
print(stem.rfm)
plot(stem.rfm)
importance(stem.rfm)
#************************#
palette_selection_cortex<- palette_selection_binary %>%
  mutate(classification = case_when(grepl("null", mix) ~ 0, #re coding each mix type (null ergot...) with a numeric integer to help make the model***
                                    grepl("glove", mix) ~ 0,
                                    grepl("cortex", mix) ~ 1,
                                    grepl("pith", mix) ~ 0)) 

cortex.rfm<- randomForest(classification~(red+green+blue),
                        data=palette_selection_cortex, 
                        ntree=40,
                        mtry = 1,
                        importance=TRUE)
print(cortex.rfm)
plot(cortex.rfm)
importance(cortex.rfm)
#************************#
palette_selection_pith<- palette_selection_circle %>%
  mutate(classification = case_when(grepl("background", mix) ~ 0, #re coding each mix type (null ergot...) with a numeric integer to help make the model***
                                    grepl("circle", mix) ~ 1,
                                    grepl("glove", mix) ~ 0)) 

pith.rfm<- randomForest(classification~(red+green+blue),
                          data=palette_selection_pith, 
                          ntree=30,
                          mtry = 1,
                          importance=TRUE)
print(pith.rfm)
plot(pith.rfm)
importance(pith.rfm)
##############
###
###
###
###



###
###
###
###
##############
## Setting file paths to import and export images and data----
#************************#
## File paths for cropped images
paths_cropped_IWG.stem<- list.files(path=folder_crop_img,full.names = TRUE)
names_cropped_IWG.stem<- list.files(path=folder_crop_img,full.names = FALSE) 
#************************#
## Folders to export images from each feature (ex. hulled stems)
## Note that these match the folders created in the beginning of the script
folder_crop_img<-  (paste(img_IWG_stem.demo.3.29.21,"crop_img",sep = "/"))
folder_glove_01<-  (paste(img_IWG_stem.demo.3.29.21,"glove_01",sep = "/"))
folder_glove_02<-  (paste(img_IWG_stem.demo.3.29.21,"glove_02",sep = "/"))
folder_stems_01<-  (paste(img_IWG_stem.demo.3.29.21,"stems_01",sep = "/"))
folder_stems_02<-  (paste(img_IWG_stem.demo.3.29.21,"stems_02",sep = "/"))
folder_stems_03<-  (paste(img_IWG_stem.demo.3.29.21,"stems_03",sep = "/"))
##############
###
###
###
###



###
###
###
###
##############
## The is the first of two for() loops for image quantification of features (ergot, seeds...)
## This loop conducts a course pixel prediction and removal of erroneous features
start<- Sys.time() #tracking time to completion***
for (i in 1:length(paths_cropped_IWG.stem)) {
  #************************#
  #************************#
  ##### Step 1
  img.crop<- readImage(paths_cropped_IWG.stem[i]) #read in the first image to use for its dimensions later on***
  img.dat.crop<- ImgToDataframe(paths_cropped_IWG.stem[i]) #vectorize the RBG array***
  img.dat.crop$classify<- predict(rfm_IWG_categorical, img.dat.crop) 
  img.dat.categorical<- img.dat.crop %>% 
    mutate(glove_pith=case_when(classify %in%  c(0,2,3) ~ 0,
                                   classify == 1 ~ 1)
           )
  #************************#
  #************************#
  ##### Step 2
  ## creating a mask or grey scale image form the predictions in the random forest model
  img.glove.1<- fillHull(matrix(img.dat.categorical$glove_pith, nrow=nrow(img.crop), ncol=ncol(img.crop)))
  
  img.glove.2<- gblur(img.glove.1, sigma = 10) > .3
  
  img.glove.3<-  fillHull(img.glove.2)
  
  disc = makeBrush(21, "disc") #setting the shape for adaptive thresholding***
  img.glove.4<- gblur(filter2(img.glove.3, disc), sigma = 5) > .9

  img.glove.5<- bwlabel(img.glove.4)

  background<- data.frame(table(img.glove.5)) %>%
    mutate(img.glove.5 = as.numeric(img.glove.5)) %>%
    filter(img.glove.5 == 1) 
  small.obj<- data.frame(table(img.glove.5)) %>%
    mutate(img.glove.5 = as.numeric(img.glove.5)) %>%
    filter(img.glove.5 > 1) %>%
    filter(!Freq == max(Freq))
  obj.interest<- rbind(background,small.obj) [[1]] -1
  select.obj = rmObjects(img.glove.5, c(obj.interest))
  
  img.glove.overlay<- ImgOverlay(img.crop,select.obj)
  img.glove.paint<- paintObjects(select.obj, img.crop, col = '#000000', thick = T)
  writeJPEG(img.glove.overlay, paste(folder_glove_01, "/", names_cropped_IWG.stem[i],"_glove_01.jpeg" ,sep=""), 
             quality = 100)
  writeImage(img.glove.paint, paste(folder_glove_02, "/", names_cropped_IWG.stem[i],"_glove_02.jpeg" ,sep=""), 
             quality = 100)
} 
end<- Sys.time() #tracking time to completion***
section_I<- end-start #tracking time to completion***
##############
###
###
###
###



###
###
###
###
##############
## Second of two for() loops
## This sections will due fine tine predictions and generate the data which analyses can be conducted 
start<- Sys.time() #tracking time to completion***
for (i in 1:length(paths_cropped_IWG.stem)) {
  img.crop<-flop(rotate(readImage(paths_cropped_IWG.stem[i]), angle = 90))
  #************************#
  #************************#
  glove_pith.dat<- ImgToDataframe(list.files(path=folder_glove_01,full.names = TRUE)[i])
  #************************#
  #************************#
  glove_pith.dat$classify<- predict(stem.rfm, glove_pith.dat)
  glove_pith.dat$thresh<- ifelse(glove_pith.dat$classify>0.80, #all pixels above 0.1 probability of chaff, global thresholding is usually a pretty good idea***
                                 glove_pith.dat$classify, 0)
  
  img.stem_01<- gblur(matrix(glove_pith.dat$thresh, nrow=nrow(img.crop), ncol=ncol(img.crop)), sigma = 1) >.4
  
  disc = makeBrush(3, "disc") #setting the shape for adaptive thresholding***
  img.stem_02<- filter2(img.stem_01, disc)
  img.stem_03<- fillHull(gblur(img.stem_02, sigma = 3) >.8)
  
  img.stem.overlay<- ImgOverlay(img.crop,img.stem_03)
  writeJPEG(img.stem.overlay, paste(folder_stems_01, "/", names_cropped_IWG.stem[i],"_stems_01.jpeg" ,sep=""), 
            quality = 100)
  #************************#
  #************************#
  stem_01.dat<- ImgToDataframe(list.files(path=folder_stems_01,full.names = TRUE)[i])
  stem_01.dat$classify<- predict(cortex.rfm, stem_01.dat)
  stem_01.dat$thresh<- ifelse(stem_01.dat$classify>0.40, #all pixels above 0.1 probability of chaff, global thresholding is usually a pretty good idea***
                              stem_01.dat$classify, 0)
  
  img.crop2<- readImage(paths_cropped_IWG.stem[i])
  img.cortex_01<- matrix(stem_01.dat$thresh, nrow=nrow(img.crop2), ncol=ncol(img.crop2))

  disc = makeBrush(7, "disc") #setting the shape for adaptive thresholding***
  img.cortex_02<- gblur(filter2(img.cortex_01, disc), sigma = 3) >.1
  img.cortex_03<- fillHull(img.cortex_02)
  
  img.cortext.overlay<- ImgOverlay(img.crop2,img.cortex_03)
  writeJPEG(img.cortext.overlay, paste(folder_stems_02, "/", names_cropped_IWG.stem[i],"_stems_02.jpeg" ,sep=""), 
            quality = 100)

}
end<- Sys.time() #tracking time to completion***
section_II<- end-start #tracking time to completion***
##############
###
###
###
###



###
###
###
###
##############
img.stats_IWG_stem.demo.3.29.21<- data.frame()


start<- Sys.time() #tracking time to completion***
for (i in 1:length(paths_cropped_IWG.stem)) {
  img.crop<-flop(rotate(readImage(paths_cropped_IWG.stem[i]), angle = 90))
  display(img.crop)
  #************************#
  #************************#
  pith.dat<- ImgToDataframe(list.files(path=folder_stems_02,full.names = TRUE)[i])
  #************************#
  #************************#
  pith.dat$classify<- predict(pith.rfm, pith.dat)
  pith.dat$thresh<- ifelse(pith.dat$classify>0.4, #all pixels above 0.1 probability of chaff, global thresholding is usually a pretty good idea***
                           pith.dat$classify, 0)
  
  img.pith_01<- matrix(pith.dat$thresh, nrow=nrow(img.crop), ncol=ncol(img.crop))
  
  
  disc = makeBrush(5, "disc")
  img.pith_02.1<- filter2(img.pith_01, disc)
  img.pith_02.2<- erode(img.pith_02.1)
  
  img.pith_03<- gblur(fillHull(img.pith_02.2), sigma = .5) > 0.99
  
  
  
  img.pith_04<- watershed(distmap(img.pith_03), ext = 15, tolerance = 2)
  display(colorLabels(img.pith_04))
  writeJPEG(colorLabels(img.pith_04), paste(folder_stems_03, "/", names_cropped_IWG.stem[i],"_stems_03.jpeg" ,sep=""), 
            quality = 100)
  
  stem.featr<- data.frame(computeFeatures.shape(img.pith_04)) %>%
    filter(s.area < 2000 & s.area > 100)
  
  write.stats<- data.frame(img.ID=              str_sub(names_cropped_IWG.stem[i]), #unique image ID***
                           stem.ct=             length(stem.featr$s.area)
                           )
  
  img.stats_IWG_stem.demo.3.29.21<-rbind(img.stats_IWG_stem.demo.3.29.21, write.stats)
}
end<- Sys.time() #tracking time to completion***
section_III<- end-start #tracking time to completion***

hist(img.stats_IWG_stem.demo.3.29.21$stem.ct)







##############
###
###
###
###



###
###
###
###
##############
## Data analysis sing the output
## This is a simple demo of how this data could be used, although many more analyses could easily be conducted
comp.dat<- img.stats_IWG_stem.demo.3.29.21 %>%
  mutate(barcode = as.numeric(substr(img.ID, 1, 4)))

vis.dat<- IWG_seed.dat %>%
  filter(year==2018 & loc == "STP") %>%
  right_join(comp.dat, by="barcode") %>%
  rename(comp_tiller_ct=stem.ct)

summary(lm(manual_tiller_ct ~ comp_tiller_ct + I(comp_tiller_ct^2), data = vis.dat))
summary(lm(manual_tiller_ct ~ comp_tiller_ct, data = vis.dat))

cor.stems<- cor.test(vis.dat$manual_tiller_ct, vis.dat$comp_tiller_ct)
regres.stems<- data.frame(comp_tiller_ct = seq(1:270),
                          manual_tiller_ct = predict(lm(manual_tiller_ct ~ comp_tiller_ct, data = vis.dat), 
                       newdata = data.frame(comp_tiller_ct = seq(1:270))))

stem_theme<- function(base_size = 18) {
  theme_minimal(base_size = base_size) %+replace%
    theme(strip.background = element_rect(fill = "grey85", color = "black", linetype = 1),
          legend.background =  element_rect(fill = "white", linetype = 0),
          legend.position = "bottom",
          panel.grid.major.y = element_line(linetype = "dashed", color = "grey80", size = .1),
          panel.grid.major.x = element_line(linetype = "dotted", color = "grey80", size = .1),
          panel.grid.minor = element_blank(),
          panel.border = element_rect(fill = alpha("white", 0), color = "grey85"),
          axis.line = element_line(size = 1.5, colour = "grey80"),
          complete = TRUE)
}


hello<- ggplot() +
  geom_point(vis.dat, 
             mapping=aes(x= comp_tiller_ct,
                 y= manual_tiller_ct,
                 group=1,
                 text= paste(
                   "<br>image_ID: ", img.ID,
                   "<br>computer count:", comp_tiller_ct,
                   "<br>manual count:", manual_tiller_ct
                 )),
             color="dodgerblue4") +
  geom_line(regres.stems, mapping = aes(x= comp_tiller_ct,
                                        y= manual_tiller_ct),
            color="dodgerblue2") +
  annotate("text", x = 50, y = 250, label = paste("r= ", round(cor.stems$estimate[[1]],2)))+
  labs(x= "Computer Predicted Stem Count",
       y= "Manual Stem Count")+
  stem_theme()
  
ggplotly(hello, tooltip = "text")


