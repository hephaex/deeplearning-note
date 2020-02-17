```
$nvidia-smi
```

```
import tourch
print(torch.cuda.is_available())

torch.cuda.get_device_name(0)
```

```
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
```
