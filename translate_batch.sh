docker compose --env-file .env -f demo/doc/docker-compose-web-with-gpu.yml up -d
# docker exec -it manga_image_translator_gpu python -m manga_translator local -v -i data/ridibooks/20250928-tmp --config-file /app/configs/gpt-kor.toml --use-gpu
# docker exec -it manga_image_translator_gpu python -m manga_translator local -v -i data/books-20250930-part1 --config-file /app/configs/gpt-kor.toml --use-gpu

# docker exec -it manga_image_translator_gpu python -m manga_translator local -v -i data/ridibooks/20250928-tmp --config-file /app/configs/book-ocr.toml --use-gpu --save-text 

docker exec -it manga_image_translator_gpu1 python -m manga_translator local -v -i data/books-20251007-part1 --config-file /app/configs/gpt-kor.toml --use-gpu
docker exec -it manga_image_translator_gpu1 python -m manga_translator local -v -i data/books-20251007-part2 --config-file /app/configs/gpt-kor.toml --use-gpu