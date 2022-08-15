.PHONY: main

main:
	time pipenv run python3 \
		-m experiment.main \
		--adv_epsilon 8 \
		--pgd_iters 100 \
		--batch_size 256 \
		--pretrain_epochs 32 \
		--lr 1e-2 \
		--swa_epochs 16 \
		--swa_lr 1e-2 \
		--swa_lr 5e-3 \
		--swa_lr 1e-3
