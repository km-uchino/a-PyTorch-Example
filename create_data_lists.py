from utils import create_data_lists

if __name__ == '__main__':
    #create_data_lists(voc07_path='/media/ssd/ssd data/VOC2007',
    #                  voc12_path='/media/ssd/ssd data/VOC2012',
    #                  output_folder='./')
    create_data_lists(voc07_path='/home/uchino/export/datasets/VOCdevkit/VOC2007',
                      voc12_path='/home/uchino/export/datasets/VOCdevkit/VOC2012',
                      output_folder='./')
