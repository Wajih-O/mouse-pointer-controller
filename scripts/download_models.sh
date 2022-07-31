# requires omz_downloader to be installed
if [ "$#" -eq 1 ];
then
    # TODO test if the models file exist
    echo "downlod models from $1"
    cat $1 | while read model;do omz_downloader --name $model -o models ; done
fi

