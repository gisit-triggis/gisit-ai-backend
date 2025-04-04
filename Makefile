commit:
	@git add .
	@git commit -m "fix"
	@git push

proto:
	@echo "Обновление прото файлов из gisit-proto..."
	rm -rf temp-proto
	@git clone --depth 1 --filter=blob:none --sparse https://github.com/gisit-triggis/gisit-proto temp-proto
	cd temp-proto && git sparse-checkout set gen/python
	cp -r temp-proto/gen/python/* generated/
	rm -rf temp-proto

PHONY: commit protoc
