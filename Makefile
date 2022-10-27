# Project setup
initialize_workspace:
	mkdir ${WORKSPACE}{metaL_data,checkpoint/label_learn/saves} -p

# downloading data
download_aircraft:
	wget https://figshare.com/ndownloader/files/37914690?private_link=a413a04a37d189638118 -O ${WORKSPACE}metaL_data/aircraft.tar.gz

download_cu_birds:
	wget https://figshare.com/ndownloader/files/37832133?private_link=3c10efda4881af51db57 -O ${WORKSPACE}metaL_data/cu_birds.tar.gz

download_vgg_flower:
	wget https://figshare.com/ndownloader/files/37832118?private_link=9c42222e62f2b6013568 -O ${WORKSPACE}metaL_data/vgg_flower.tar.gz

download_miniimagenet:
	wget https://figshare.com/ndownloader/files/37911240?private_link=ab81ddf3fd85f775cc6d -O ${WORKSPACE}metaL_data/miniImageNet.tar.gz

download_mini60:
	wget https://figshare.com/ndownloader/files/37835829?private_link=fc867135b430db59ceaf -O ${WORKSPACE}metaL_data/mini60.tar.gz

download_tieredimagenet:
	wget https://figshare.com/ndownloader/files/38009146?private_link=a7afdc3fb808064be581 -O ${WORKSPACE}metaL_data/tieredImageNet.tar.gz

download_tiered780:
	wget https://figshare.com/ndownloader/files/37836714?private_link=89a52d40c31da8724b61 -O ${WORKSPACE}metaL_data/tiered780.tar.gz

download_all: download_aircraft download_cu_birds download_vgg_flower download_miniimagenet download_mini60 download_tieredimagenet download_tiered780

unpack_datasets:
	for file in ${WORKSPACE}metaL_data/* ; do \
		tar -xzvf $$file -C ${WORKSPACE}metaL_data ; \
	done
