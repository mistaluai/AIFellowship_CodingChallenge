mkdir -p data/images && wget -O data/102flowers.tgz https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz && tar -xzf data/102flowers.tgz -C data/images && rm data/102flowers.tgz
wget -O data/imagelabels.mat https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat
wget -O data/setid.mat https://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat