import os
from absl import flags
import sys
from subprocess import run

import object_detection
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format

# Обрабатываем флаги
flags.DEFINE_integer('steps', 2000, 'Number of train steps.', short_name='s')
flags.DEFINE_bool('eval_model', False, 'Evaluate model or not', short_name='e')
flags.DEFINE_bool('verificate', True, 'Verificate model or not', short_name='ve')
FLAGS = flags.FLAGS
FLAGS(sys.argv)

def main():
    # Задаем пути
    CUSTOM_MODEL_NAME = 'my_ssd_mobnet' 
    PRETRAINED_MODEL_NAME = 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'
    PRETRAINED_MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz'
    TF_RECORD_SCRIPT_NAME = 'generate_tfrecord.py'
    LABEL_MAP_NAME = 'label_map.pbtxt'

    paths = {
        'WORKSPACE_PATH': os.path.join('Tensorflow', 'workspace'),
        'SCRIPTS_PATH': os.path.join('Tensorflow','scripts'),
        'APIMODEL_PATH': os.path.join('Tensorflow','models'),
        'ANNOTATION_PATH': os.path.join('Tensorflow', 'workspace','annotations'),
        'IMAGE_PATH': os.path.join('Tensorflow', 'workspace','images'),
        'MODEL_PATH': os.path.join('Tensorflow', 'workspace','models'),
        'PRETRAINED_MODEL_PATH': os.path.join('Tensorflow', 'workspace','pre-trained-models'),
        'CHECKPOINT_PATH': os.path.join('Tensorflow', 'workspace','models', CUSTOM_MODEL_NAME), 
        'OUTPUT_PATH': os.path.join('Tensorflow', 'workspace','models', CUSTOM_MODEL_NAME, 'export'), 
        'TFJS_PATH':os.path.join('Tensorflow', 'workspace','models', CUSTOM_MODEL_NAME, 'tfjsexport'), 
        'TFLITE_PATH':os.path.join('Tensorflow', 'workspace','models', CUSTOM_MODEL_NAME, 'tfliteexport'), 
        'PROTOC_PATH':os.path.join('Tensorflow','protoc')
     }
    files = {
        'PIPELINE_CONFIG':os.path.join('Tensorflow', 'workspace','models', CUSTOM_MODEL_NAME, 'pipeline.config'),
        'TF_RECORD_SCRIPT': os.path.join(paths['SCRIPTS_PATH'], TF_RECORD_SCRIPT_NAME), 
        'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME)
    }
    # Проверяем, что с моделью все окей - на выходе должны получить 'OK (skipped=1)'
    if FLAGS.verificate:
        VERIFICATION_SCRIPT = os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection', 'builders', 'model_builder_tf2_test.py')
        cmd = 'python ' + VERIFICATION_SCRIPT
        run(cmd.split())

    # Записываем метки
    labels = [{'name':'pena', 'id':1}, {'name':'fontan', 'id':2}, {'name':'shapka', 'id':3}]
    with open(files['LABELMAP'], 'w') as f:
        for label in labels:
            f.write('item { \n')
            f.write('\tname:\'{}\'\n'.format(label['name']))
            f.write('\tid:{}\n'.format(label['id']))
            f.write('}\n')
    print('Successfully created labels')

    # Cоздаем TFrecord файлы
    cmd = f"python {files['TF_RECORD_SCRIPT']} -x {os.path.join(paths['IMAGE_PATH'], 'train')} -l {files['LABELMAP']} -o {os.path.join(paths['ANNOTATION_PATH'], 'train.record')}"
    run(cmd.split())
    cmd = f"python {files['TF_RECORD_SCRIPT']} -x {os.path.join(paths['IMAGE_PATH'], 'test')} -l {files['LABELMAP']} -o {os.path.join(paths['ANNOTATION_PATH'], 'test.record')}"
    run(cmd.split())

    # Загружаем настройки модели
    config = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])

    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.io.gfile.GFile(files['PIPELINE_CONFIG'], "r") as f:                                                                                                                                                                                                                     
        proto_str = f.read()                                                                                                                                                                                                                                          
        text_format.Merge(proto_str, pipeline_config)  

    pipeline_config.model.ssd.num_classes = len(labels)
    pipeline_config.train_config.batch_size = 4
    pipeline_config.train_config.fine_tune_checkpoint = os.path.join(paths['PRETRAINED_MODEL_PATH'], PRETRAINED_MODEL_NAME, 'checkpoint', 'ckpt-0')
    pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
    pipeline_config.train_input_reader.label_map_path= files['LABELMAP']
    pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [os.path.join(paths['ANNOTATION_PATH'], 'train.record')]
    pipeline_config.eval_input_reader[0].label_map_path = files['LABELMAP']
    pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [os.path.join(paths['ANNOTATION_PATH'], 'test.record')]

    config_text = text_format.MessageToString(pipeline_config)                                                                                                                                                                                                        
    with tf.io.gfile.GFile(files['PIPELINE_CONFIG'], "wb") as f:                                                                                                                                                                                                                     
        f.write(config_text) 
    print('Successfully downloaded config') 

    # Обучение модели
    TRAINING_SCRIPT = os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection', 'model_main_tf2.py')
    cmd = f"python {TRAINING_SCRIPT} --model_dir={paths['CHECKPOINT_PATH']} --pipeline_config_path={files['PIPELINE_CONFIG']} --num_train_steps={FLAGS.steps}"
    run(cmd.split())
    print('Successfully train model') 

    # Фризим веса
    FREEZE_SCRIPT = os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection', 'exporter_main_v2.py ')
    cmd = f"python {FREEZE_SCRIPT} --input_type=image_tensor --pipeline_config_path={files['PIPELINE_CONFIG']} --trained_checkpoint_dir={paths['CHECKPOINT_PATH']} --output_directory={paths['OUTPUT_PATH']}"
    run(cmd.split())
    print('Successfully freezed graph')

    # Конвертируем модель
    TFLITE_SCRIPT = os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection', 'export_tflite_graph_tf2.py ')
    cmd = f"python {TFLITE_SCRIPT} --pipeline_config_path={files['PIPELINE_CONFIG']} --trained_checkpoint_dir={paths['CHECKPOINT_PATH']} --output_directory={paths['TFLITE_PATH']}"
    run(cmd.split())

    FROZEN_TFLITE_PATH = os.path.join(paths['TFLITE_PATH'], 'saved_model')
    TFLITE_MODEL = os.path.join(paths['TFLITE_PATH'], 'saved_model', 'detect.tflite')
    cmd = f"tflite_convert \
        --saved_model_dir={FROZEN_TFLITE_PATH} \
        --output_file={TFLITE_MODEL} \
        --input_shapes=1,300,300,3 \
        --input_arrays=normalized_input_image_tensor \
        --output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3' \
        --inference_type=FLOAT \
        --allow_custom_ops"
    run(cmd.split())
    print('Successfully converted model')

    # Загружаем архив в корень
    cmd = f"tar -czf models.tar.gz {paths['CHECKPOINT_PATH']}"
    run(cmd.split())
    print('Successfully exported model')

    # Тестирование модели
    if FLAGS.eval_model:
        cmd = f"python {TRAINING_SCRIPT} --model_dir={paths['CHECKPOINT_PATH']} --pipeline_config_path={files['PIPELINE_CONFIG']} --checkpoint_dir={paths['CHECKPOINT_PATH']}"
        run(cmd.split())
        print('Successfully evaluate model')

if __name__ == '__main__':
    main()
