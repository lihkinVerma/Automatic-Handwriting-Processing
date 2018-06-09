f<- read.csv('test.csv',header=FALSE)
im=f[10,]
tag<-im[1]
im<- im[2:length(im)]

z<-c(1:28)
for (i in seq(1,784,by=28) ){
  j=i+27
  names(z)<- names(im[i:j])
  z<- rbind(z,im[i:j])
}
image(as.matrix(z[2:29,]))

library(imager)
fpath <- system.file('C:/Users/Nikhil/Desktop/finalized/5. Working model/hey.png',package='imager')
parrots <- load.image(fpath)
plot(parrots)

library(spatstat)
a<- as.im(im)
