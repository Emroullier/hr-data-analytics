run:
	@python hr-data-analytics-package-folder/main.py

run_uvicorn:
	@uvicorn hr-data-analytics-package-folder.api:app --reload

install:
	@pip install -e .

test:
	@pytest -v tests
