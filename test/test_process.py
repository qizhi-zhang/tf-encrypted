from multiprocessing import Process


def f(x):
    for i in range(10000000000):
        x=x/0.0
    return x

p = Process(target=f, args=(1,))
        #status=_start_server(task_id, XOwner_iphost, YOwner_iphost, RS_iphost, Player)
p.start()


p.join(4)
print(p.exitcode)
if p.exitcode==0:
    print(p.exitcode)
else:
    print(-1)


