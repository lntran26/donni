test:
	pytest -xv --flake8 .
	pylint tests

test_generate_data:
	pytest -xv --flake8 test_generate_data.py
	pylint test_generate_data.py
	pylint generate_data.py

test_train:
	pytest -xv --flake8 test_train.py
	pylint test_train.py
	pylint train.py

test_integration:
	pytest -xv --flake8 test_integration.py
	pylint test_integration.py
	pylint __main__.py

test_predict:
	pytest -xv --flake8 test_predict.py
	pylint test_predict.py
	pylint predict.py