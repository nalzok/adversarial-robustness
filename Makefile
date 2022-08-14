.PHONY: main

main:
	time pipenv run python3 \
		-m experiment.main \
		--adv_epsilon 8 \
		--pgd_iters 200 \
		--batch_size 256 \
		--epochs 64 \
		--lr 1e-2 \
		--swa_start 48 \
		--swa_lr 5e-3
