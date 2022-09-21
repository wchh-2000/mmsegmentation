from matplotlib import pyplot as plt
import jsonlines
def show_train(dir,fname,val=False,interval=50,showname=None):     
    loss=[]
    loss_val=[]
    lr=[]
    d_acc=[]
    d_acc_v=[]
    a_acc=[]
    a_acc_v=[]
    t=[]
    if val:        
        with jsonlines.open(dir+fname,'r') as f:
            for d in f:
                if 'loss_val' not in d:
                    continue
                loss.append(d['loss'])
                loss_val.append(d['loss_val'])

                d_acc.append(d['decode.acc_seg'])
                d_acc_v.append(d['decode.acc_seg_val'])
                a_acc.append(d['aux.acc_seg'])
                a_acc_v.append(d['aux.acc_seg_val'])

                #lr.append(d['lr'])
        x=list(range(len(loss)))
        x=[interval*n for n in x]
        plt.plot(x,loss,label='loss')
        plt.plot(x,loss_val,label='loss_val')
        plt.legend()
        plt.savefig(dir+"loss.png")
        # plt.clf()
        # plt.plot(x,lr)
        # plt.savefig(dir+'lr.png')
        plt.clf()
        label_lst=['decode.acc_seg','decode.acc_seg_val','aux.acc_seg','aux.acc_seg_val']
        for y,l in zip([d_acc,d_acc_v,a_acc,a_acc_v],label_lst):
            plt.plot(x,y,label=l)
        plt.legend()
        plt.savefig(dir+'acc.png')
    else:#no val
        with jsonlines.open(dir+fname,'r') as f:
            for d in f:
                if 'loss' not in d:
                    continue
                loss.append(d['loss'])
                d_acc.append(d['decode.acc_seg'])
                a_acc.append(d['aux.acc_seg'])
                if showname:
                    t.append(d[showname])
                # lr.append(d['lr'])
        x=list(range(len(loss)))
        x=[interval*n for n in x]
        plt.plot(x,loss,label='loss_ce')
        if showname:
            plt.plot(x,t,label=showname)
            plt.legend()
        plt.savefig(dir+"loss.png")

        # plt.clf()
        # plt.plot(x,lr)
        # plt.savefig(dir+'lr.png')
        plt.clf()
        plt.plot(x,a_acc,label='aux.acc_seg')
        plt.plot(x,d_acc,label='decode.acc_seg')
        plt.legend()
        plt.savefig(dir+'acc.png')

show_train("work_dirs/convnext_l_lova_aug/","20220920_145634.log.json",val=False \
    ,interval=50)#,showname="decode.loss_lova")
def compare(dir,f1,f2,interval=50):
    acc0=[]
    acc1=[]
    with jsonlines.open(dir+f1,'r') as f:
        for d in f:
            if 'loss' not in d:
                continue
            acc0.append(d['decode.acc_seg'])
    with jsonlines.open(dir+f2,'r') as f:
        for d in f:
            if 'loss' not in d:
                continue
            acc1.append(d['decode.acc_seg'])
    x=list(range(len(acc0)))
    x=[interval*n for n in x]

    plt.plot(x,acc0,label='decode.acc_base')
    plt.plot(x,acc1,label='decode.acc_dice')
    plt.legend()
    plt.savefig(dir+'compare_acc.png')
#compare("work_dirs/loss_test/","base/20220914_095332.log.json","dice/20220914_095332.log.json")
# pth="test_results/0.png"
# from PIL import Image
# import numpy as np
# img=np.array(Image.open(pth))
# print(img.size)