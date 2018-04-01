### 第三章
字典中的元素必须是可以散列的，如列表就是不可散列的。

**字典推导**
```py
country_code={country:code for code,country in DIAL_CODES}
# 后者为一个元组的列表

country_code={country.upper():code for code,country in DIAL_CODES
                                                      if code<66}
# 也可以加入筛选条件
```

**常见的映射方法**
```py
d.__delitem__(k) # del k 时调用
d.get(k,[default]) # 返回键k对应的值，如果没找到就返回default
d.__getitem__(k) # 实现【】
d.keys() # 返回所有的键
d.setdefault(k,[default]) # 字典中键k设为default，但不改变原来的值，没有则创建一个
d.setitem__(k,v) # 实现d[k]=v
d.update(m,[**kargs]) # 如果m是键值对迭代器则直接更新，如果不是则看成映射
d.values() # 返回值
```

用get方法或是 `d[k]` 时，如果没有元素会报错，这时一般用 `d.get(k,default)` 来代替，给找不到的键一个默认的返回值。
```py
index.setdefault(word,[]).append(location)
# setdefault如果没找到返回[]，找到返回找到的值，不会改变原来的值
```

**defaultdict**
处理找不到的键，可以用这个，在创建defaultdict对象的时候就创建了处理找不到的键的方法。
```py
index=collections.defaultdict(list) # 参数是一个list构造方法
```

**集合**
```py
found=len(needles & haystack)
# 运算的二者都是集合，集合求交来求公共部分的数量快速而简洁

found=len(set(needles).intersection(haystack))
# 也可以这样写，只要有一方是集合，速度就快

set()
# 空集必须这样生成，用{}会生成dict

{chr(i) for i in range(32,256) if 'SIGN' in name(chr(i),'')}
# 集合推导的写法
```

**集合操作**
```py
s|=z
s.update(it,...) # 和上面的语句相等，集合还支持很多其他运算符

s.isdisjoint(z) # 不相交
e in s # 属于
s<=z # 子集
s<z # 真子集
s>z # 同上
s>=z # 同上
```
**不要一个一个添加，应该全部生成之后，再一次update**
