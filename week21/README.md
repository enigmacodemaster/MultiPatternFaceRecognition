### 改动
1. 传入的anc_img加上了cuda()
2. str(anc_img) ==> str(anc_img, encoding='utf-8')
3. 对resnet再稍作修改适应传入图片前向传播之后的tensor size
```
self.model.fc = nn.Linear(input_features_fc_layer, embedding_dimension)
```
改为
```
        self.model.fc = nn.Sequential(
        		nn.Linear(2048, input_features_fc_layer),
        		nn.Linear(input_features_fc_layer, embedding_dimension)
        )
``` 
否则，报错，tensorsize不匹配
