.PHONY: main

main:
	time pipenv run python3 \
		-m experiment.main \
		--adv_epsilon 8 \
		--pgd_iters 500 \
		--batch_size 256 \
		--pretrain_epochs 64 \
		--lr 5e-3 \
		--swa_epochs 64 \
		--swa_lr 5e-3 \
		--swa_lr 4e-3 \
		--swa_lr 3e-3 \
		--swa_lr 2e-3 \
		--swa_lr 1e-3
