run:
	@python hr_data_analytics_package_folder/main.py

run_uvicorn:
	@uvicorn hr_data_analytics_package_folder.api:app --reload

install:
	@pip install -e .

test:
	@pytest -v tests


reset_dummy_data_files:
	@rm -rf ${DUMMY_DIR}
	@mkdir  ${DUMMY_DIR}
