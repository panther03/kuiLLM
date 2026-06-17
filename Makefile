infer: infer-kuiper
infer-kuiper:
	python3 infer.py

infer-no-kuiper:
	python3 infer.py --no-kuiper

profile-kuiper-calls:
	PRINT_PROFILING=1 python3 infer.py

profile-no-kuiper-calls:
	PRINT_PROFILING=1 python3 infer.py --no-kuiper

profile-kuiper-nsys:
	nsys profile -o data/kuiper.nsys-rep --force-overwrite=true -t cuda python3 infer.py

profile-no-kuiper-nsys:
	nsys profile -o data/no-kuiper.nsys-rep --force-overwrite=true -t cuda python3 infer.py --no-kuiper

test:
	python3 -m pytest tests/