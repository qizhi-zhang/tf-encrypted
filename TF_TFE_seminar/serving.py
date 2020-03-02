import os




server_cmd="""
docker run -dt -p 8501:8501 -v "$(pwd)/save_model:/models/test_lr" -e MODEL_NAME=test_lr tensorflow/serving
"""

os.system(server_cmd)



clint_cmd="""
curl -d '{"inputs": [[1.0, 2.0, 5.0]]}' -X POST http://localhost:8501/v1/models/test_lr:predict
"""









"""
docker exec -it 62c9c5a5093e     /bin/bash
"""

"""
tensorflow_model_server --port=8500 --rest_api_port=8501   --model_name=test_lr --model_base_path=/models/test_lr
"""
