import paddle
def gram_matrix(feat):
    # https://github.com/pytorch/examples/blob/master/fast_neural_style/neural_style/utils.py
    (b, ch, h, w) = feat.shape
    # feat = feat.view(b, ch, h * w)
    feat = paddle.reshape(feat, [b, ch, h*w])
    feat_t = paddle.transpose(feat, perm=[0, 2, 1])
    gram = paddle.bmm(feat, feat_t) / (ch * h * w) #torch.bmm(a,b)，a和b进行矩阵的乘法，且a，b都是三维的tensor，必须满足a是(b,h,w)b是(b,w,h)
    return gram

if __name__=='__main__':
    a=paddle.randn([1,2,4,4])
    print(a)
    b=gram_matrix(a)
    print(b)