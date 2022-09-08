import hub
ds = hub.load("hub://activeloop/ffhq")[20000]['images_1024/image']
print(ds)