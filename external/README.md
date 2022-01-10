# How To Use External Model
## 1. Clone from remote repository
```
cd ./external/
git clone [repository.git link]
```

## 2. Download checkpoint
download checkpoint .pth file into <code>./result/weight/</code>

## 3. Add importlib code
Fix <code>get_model</code> function from <code>./utils/models.py</code>

(Optional) Fix <code>get_finetune_optimizer</code> function from <code>./utils/optims.py</code>


