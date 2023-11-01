import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.init import xavier_uniform_
import contrastive_loss as cl


class Scholar(object):

    def __init__(self, config, alpha=1.0, learning_rate=0.001, init_embeddings=None, update_embeddings=True,
                 init_bg=None, update_background=True, adam_beta1=0.99, adam_beta2=0.999, device=None, seed=None,
                 classify_from_covars=True):

        """
        Create the model
        :param config: a dictionary with the model configuration
        :param alpha: hyperparameter for the document representation prior
        :param learning_rate: learning rate for Adam
        :param init_embeddings: a matrix of embeddings to initialize the first layer of the bag-of-words encoder
        :param update_embeddings: if True, update word embeddings during training
        :param init_bg: a vector of empirical log backgound frequencies
        :param update_background: if True, update the background term during training
        :param adam_beta1: first hyperparameter for Adam
        :param adam_beta2: second hyperparameter for Adam
        :param device: (int) the number of the GPU to use
        """

        if seed is not None:
            torch.manual_seed(seed)
            torch.backends.cudnn.deterministic = True

        self.network_architecture = config
        self.learning_rate = learning_rate
        self.adam_beta1 = adam_beta1

        self.update_embeddings = update_embeddings
        self.update_background = update_background

        # create priors on the hidden state
        self.n_topics = (config["n_topics"])

        if device is None:
            self.device = 'cpu'
        else:
            self.device = 'cuda:' + str(device)

        # interpret alpha as either a (symmetric) scalar prior or a vector prior
        if np.array(alpha).size == 1:
            # if alpha is a scalar, create a symmetric prior vector
            self.alpha = alpha * np.ones((1, self.n_topics)).astype(np.float32) # [1 * 主题数量]
        else:
            # otherwise use the prior as given
            self.alpha = np.array(alpha).astype(np.float32)
            assert len(self.alpha) == self.n_topics

        # create the pyTorch model 得到了模型，重要是得到了网络架构
        self._model = torchScholar(config, self.alpha, update_embeddings, init_emb=init_embeddings, bg_init=init_bg, device=self.device, classify_from_covars=classify_from_covars).to(self.device)

        # set the criterion 损失函数 二分类交叉熵
        self.criterion = nn.BCEWithLogitsLoss()

        # create the optimizer
        grad_params = filter(lambda p: p.requires_grad, self._model.parameters()) # 只有grad_params为true的网络才需要更新参数
        self.optimizer = optim.Adam(grad_params, lr=learning_rate, betas=(adam_beta1, adam_beta2))

    def fit(self, contrastive, merge, X, pos_X, neg_X, struct_X, Y, PC, TC, eta_bn_prop=1.0, l1_beta=None, l1_beta_c=None, l1_beta_ci=None):
        """
        Fit the model to a minibatch of data
        :param X: np.array of document word counts [batch size x vocab size]
        :param Y: np.array of labels [batch size x n_labels]
        :param PC: np.array of prior covariates influencing the document-topic prior [batch size x n_prior_covars]
        :param TC: np.array of topic covariates to be associated with topical deviations [batch size x n_topic_covars]
        :param l1_beta: np.array of prior variances on the topic weights
        :param l1_beta_c: np.array of prior variances on the weights for topic covariates
        :param l1_beta_ci: np.array of prior variances on the weights for topic-covariate interactions
        :return: loss; label pred probs; document representations; neg-log-likelihood; KLD
        """
        # move data to device
        X = torch.Tensor(X).to(self.device)
        pos_X = torch.Tensor(pos_X).to(self.device)
        neg_X = torch.Tensor(neg_X).to(self.device)
        struct_X = torch.Tensor(struct_X).to(self.device)
        # if Y is not None:
        #     Y = torch.Tensor(Y).to(self.device)
        # if PC is not None:
        #     PC = torch.Tensor(PC).to(self.device)
        # if TC is not None:
        #     TC = torch.Tensor(TC).to(self.device)
        self.optimizer.zero_grad()

        # do a forward pass 前向传播阶段 返回theta, X_recon, Y_recon, losses->[loss.mean(), NL.mean(), KLD.mean()]
        thetas, X_recon, Y_probs, losses = self._model(contrastive, merge, X, pos_X, neg_X, struct_X, Y, PC, TC, eta_bn_prop=eta_bn_prop, l1_beta=l1_beta, l1_beta_c=l1_beta_c, l1_beta_ci=l1_beta_ci)
        if contrastive:
            loss, nl, kld, contrastive_loss = losses
        else:
            loss, nl, kld = losses
        
        # update model
        loss.backward()
        self.optimizer.step()

        # if Y_probs is not None:
        #     Y_probs = Y_probs.to('cpu').detach().numpy() 返回loss, Y_probs, thetas, nl, kld
        if contrastive:
            return loss.to('cpu').detach().numpy(), Y_probs, thetas.to('cpu').detach().numpy(), nl.to('cpu').detach().numpy(), kld.to('cpu').detach().numpy(), contrastive_loss.to('cpu').detach().numpy()
        else:
            return loss.to('cpu').detach().numpy(), Y_probs, thetas.to('cpu').detach().numpy(), nl.to('cpu').detach().numpy(), kld.to('cpu').detach().numpy()
        
    def predict(self, X, PC, TC, eta_bn_prop=0.0):
        """
        Predict labels for a minibatch of data
        """
        # input a vector of all zeros in place of the labels that the model has been trained on
        batch_size = self.get_batch_size(X)
        Y = np.zeros((batch_size, self.network_architecture['n_labels'])).astype('float32')
        X = torch.Tensor(X).to(self.device)
        Y = torch.Tensor(Y).to(self.device)
        if PC is not None:
            PC = torch.Tensor(PC).to(self.device)
        if TC is not None:
            TC = torch.Tensor(TC).to(self.device)
        theta, _, Y_recon, _ = self._model(X, Y, PC, TC, do_average=False, var_scale=0.0, eta_bn_prop=eta_bn_prop)
        return theta, Y_recon.to('cpu').detach().numpy()

    def predict_from_topics(self, theta, PC, TC, eta_bn_prop=0.0):
        """
        Predict label probabilities from each topic
        """
        theta = torch.Tensor(theta)
        if PC is not None:
            PC = torch.Tensor(PC)
        if TC is not None:
            TC = torch.Tensor(TC)
        probs = self._model.predict_from_theta(theta, PC, TC)
        return probs.to('cpu').detach().numpy()

    def get_losses(self, X, Y, PC, TC, eta_bn_prop=0.0, n_samples=0):
        """
        Compute and return the loss values for all instances in X, Y, PC, and TC averaged over multiple samples
        """
        batch_size = self.get_batch_size(X)
        if batch_size == 1:
            X = np.expand_dims(X, axis=0)
        if Y is not None and batch_size == 1:
            Y = np.expand_dims(Y, axis=0)
        if PC is not None and batch_size == 1:
            PC = np.expand_dims(PC, axis=0)
        if TC is not None and batch_size == 1:
            TC = np.expand_dims(TC, axis=0)
        X = torch.Tensor(X).to(self.device)
        if Y is not None:
            Y = torch.Tensor(Y).to(self.device)
        if PC is not None:
            PC = torch.Tensor(PC).to(self.device)
        if TC is not None:
            TC = torch.Tensor(TC).to(self.device)
        if n_samples == 0:
            _, _, _, temp = self._model(X, Y, PC, TC, do_average=False, var_scale=0.0, eta_bn_prop=eta_bn_prop)
            loss, NL, KLD = temp
            losses = loss.to('cpu').detach().numpy()
        else:
            _, _, _, temp = self._model(X, Y, PC, TC, do_average=False, var_scale=1.0, eta_bn_prop=eta_bn_prop)
            loss, NL, KLD = temp
            losses = loss.to('cpu').detach().numpy()
            for s in range(1, n_samples):
                _, _, _, temp = self._model(X, Y, PC, TC, do_average=False, var_scale=1.0, eta_bn_prop=eta_bn_prop)
                loss, NL, KLD = temp
                losses += loss.to('cpu').detach().numpy()
            losses /= float(n_samples)

        return losses

    def compute_theta(self, contrastive, merge, X, struct_X, Y, PC, TC, eta_bn_prop=0.0):
        """
        Return the latent document representation (mean of posterior of theta) for a given batch of X, Y, PC, and TC
        """
        batch_size = self.get_batch_size(X)
        if batch_size == 1:
            X = np.expand_dims(X, axis=0)
        
        if batch_size == 1:
            struct_X = np.expand_dims(struct_X, axis=0)
        if Y is not None and batch_size == 1:
            Y = np.expand_dims(Y, axis=0)
        if PC is not None and batch_size == 1:
            PC = np.expand_dims(PC, axis=0)
        if TC is not None and batch_size == 1:
            TC = np.expand_dims(TC, axis=0)

        X = torch.Tensor(X).to(self.device)
        
        struct_X = torch.Tensor(struct_X).to(self.device)
        if Y is not None:
            Y = torch.Tensor(Y).to(self.device)
        if PC is not None:
            PC = torch.Tensor(PC).to(self.device)
        if TC is not None:
            TC = torch.Tensor(TC).to(self.device)
        merge_theta, _, _ = self._model(contrastive, merge, X, None, None, struct_X, Y, PC, TC, compute_loss = False, do_average=False, var_scale=0.0, eta_bn_prop=eta_bn_prop)

        return merge_theta.to('cpu').detach().numpy()

    def get_weights(self):
        """
        Return the topic-vocabulary deviation weights
        """
        emb = self._model.beta_layer.to('cpu').weight.detach().numpy().T
        self._model.beta_layer.to(self.device)
        return emb

    def get_bg(self):
        """
        Return the background terms
        """
        bg = self._model.beta_layer.to('cpu').bias.detach().numpy()
        self._model.beta_layer.to(self.device)
        return bg

    def get_prior_weights(self):
        """
        Return the weights associated with the prior covariates
        """
        emb = self._model.prior_covar_weights.to('cpu').weight.detach().numpy().T
        self._model.prior_covar_weights.to(self.device)
        return emb

    def get_covar_weights(self):
        """
        Return the topic weight (deviations) associated with the topic covariates
        """
        emb = self._model.beta_c_layer.to('cpu').weight.detach().numpy().T
        self._model.beta_c_layer.to(self.device)
        return emb

    def get_covar_interaction_weights(self):
        """
        Return the weights (deviations) associated with the topic-covariate interactions
        """
        emb = self._model.beta_ci_layer.to('cpu').weight.detach().numpy().T
        self._model.beta_ci_layer.to(self.device)
        return emb

    def get_batch_size(self, X):
        """
        Get the batch size for a minibatch of data
        :param X: the minibatch
        :return: the size of the minibatch
        """
        if len(X.shape) == 1:
            batch_size = 1
        else:
            batch_size, _ = X.shape
        return batch_size

    def eval(self):
        self._model.eval()

    def train(self):
        self._model.train()


class torchScholar(nn.Module):

    def __init__(self, config, alpha, update_embeddings=True, init_emb=None, bg_init=None, device='cpu', classify_from_covars=False):
        super(torchScholar, self).__init__()

        # load the configuration
        self.vocab_size = config['vocab_size']
        self.words_emb_dim = config['embedding_dim']
        self.n_topics = config['n_topics']
        self.n_labels = config['n_labels']
        self.n_prior_covars = config['n_prior_covars']
        self.n_topic_covars = config['n_topic_covars']
        self.classifier_layers = config['classifier_layers']
        self.use_interactions = config['use_interactions']
        self.l1_beta_reg = config['l1_beta_reg']
        self.l1_beta_c_reg = config['l1_beta_c_reg']
        self.l1_beta_ci_reg = config['l1_beta_ci_reg']
        self.l2_prior_reg = config['l2_prior_reg']
        self.device = device
        self.classify_from_covars = classify_from_covars

        # None!!! create a layer for prior covariates to influence the document prior
        if self.n_prior_covars > 0:
            self.prior_covar_weights = nn.Linear(self.n_prior_covars, self.n_topics, bias=False)
        else:
            self.prior_covar_weights = None

        # create the encoder
        self.embeddings_x_layer = nn.Linear(self.vocab_size, self.words_emb_dim, bias=False) # 向量：词表长度->words_emb_dim
        emb_size = self.words_emb_dim
        classifier_input_dim = self.n_topics
        if self.n_prior_covars > 0:
            emb_size += self.n_prior_covars
            if self.classify_from_covars:
                classifier_input_dim += self.n_prior_covars
        if self.n_topic_covars > 0:
            emb_size += self.n_topic_covars
            if self.classify_from_covars:
                classifier_input_dim += self.n_topic_covars
        if self.n_labels > 0:
            emb_size += self.n_labels

        self.encoder_dropout_layer = nn.Dropout(p=0.2) # dropout层

        if not update_embeddings: # 要更新embedding，跳过下一句
            self.embeddings_x_layer.weight.requires_grad = False
        if init_emb is not None:
            self.embeddings_x_layer.weight.data.copy_(torch.from_numpy(init_emb)).to(self.device)
        else:
            xavier_uniform_(self.embeddings_x_layer.weight) # 随机初始化encoder参数

        # create the mean and variance components of the VAE
        self.mean_layer = nn.Linear(emb_size, self.n_topics) # emb_size -> topic_size 生成潜在变量的均值部分
        self.logvar_layer = nn.Linear(emb_size, self.n_topics) # emb_size -> topic_size 生成潜在变量的对数方差部分

        self.mean_bn_layer = nn.BatchNorm1d(self.n_topics, eps=0.001, momentum=0.001, affine=True) # 对均值进行批归一化
        self.mean_bn_layer.weight.data.copy_(torch.from_numpy(np.ones(self.n_topics))).to(self.device) # 对bn层权重初始化为1
        self.mean_bn_layer.weight.requires_grad = False # 不更新
        self.logvar_bn_layer = nn.BatchNorm1d(self.n_topics, eps=0.001, momentum=0.001, affine=True)
        self.logvar_bn_layer.weight.data.copy_(torch.from_numpy(np.ones(self.n_topics))).to(self.device)
        self.logvar_bn_layer.weight.requires_grad = False

        self.z_dropout_layer = nn.Dropout(p=0.2) # dropout层

        # create the decoder
        self.beta_layer = nn.Linear(self.n_topics, self.vocab_size) # 向量: topic_size -> vocab_size

        xavier_uniform_(self.beta_layer.weight) # 随机初始化向量
        if bg_init is not None: # 将之前计算的每个词的词频记为解码器的偏置项
            self.beta_layer.bias.data.copy_(torch.from_numpy(bg_init))
            self.beta_layer.bias.requires_grad = False
        self.beta_layer = self.beta_layer.to(self.device)

        # if self.n_topic_covars > 0:
        #     self.beta_c_layer = nn.Linear(self.n_topic_covars, self.vocab_size, bias=False).to(self.device)
        #     if self.use_interactions:
        #         self.beta_ci_layer = nn.Linear(self.n_topics * self.n_topic_covars, self.vocab_size, bias=False).to(self.device)

        # create the classifier
        # if self.n_labels > 0:
        #     if self.classifier_layers == 0:
        #         self.classifier_layer_0 = nn.Linear(classifier_input_dim, self.n_labels).to(self.device)
        #     else:
        #         self.classifier_layer_0 = nn.Linear(classifier_input_dim, classifier_input_dim).to(self.device)
        #         self.classifier_layer_1 = nn.Linear(classifier_input_dim, self.n_labels).to(self.device)

        # create a final batchnorm layer 对最后解码得到的向量进行批归一化
        self.eta_bn_layer = nn.BatchNorm1d(self.vocab_size, eps=0.001, momentum=0.001, affine=True).to(self.device)
        self.eta_bn_layer.weight.data.copy_(torch.from_numpy(np.ones(self.vocab_size)).to(self.device))
        self.eta_bn_layer.weight.requires_grad = False

        # create the document prior terms 文档的先验分布
        prior_mean = (np.log(alpha).T - np.mean(np.log(alpha), 1)).T
        prior_var = (((1.0 / alpha) * (1 - (2.0 / self.n_topics))).T + (1.0 / (self.n_topics * self.n_topics)) * np.sum(1.0 / alpha, 1)).T

        prior_mean = np.array(prior_mean).reshape((1, self.n_topics))
        prior_logvar = np.array(np.log(prior_var)).reshape((1, self.n_topics))
        self.prior_mean = torch.from_numpy(prior_mean).to(self.device)
        self.prior_mean.requires_grad = False
        self.prior_logvar = torch.from_numpy(prior_logvar).to(self.device)
        self.prior_logvar.requires_grad = False

        # transformer层融合网络
        self.topic2emb = nn.Linear(self.n_topics, emb_size)
        self.transformer_layer = nn.Transformer(d_model=emb_size, nhead=2, num_encoder_layers=2, num_decoder_layers=2).to(self.device)

    def forward(self, contrastive, merge, X, pos_X, neg_X, struct_X, Y, PC, TC, compute_loss=True, do_average=True, eta_bn_prop=1.0, var_scale=1.0, l1_beta=None, l1_beta_c=None, l1_beta_ci=None):
        """
        Do a forward pass of the model
        :param X: np.array of word counts [batch_size x vocab_size]
        :param Y: np.array of labels [batch_size x n_classes]
        :param PC: np.array of covariates influencing the prior [batch_size x n_prior_covars]
        :param TC: np.array of covariates with explicit topic deviations [batch_size x n_topic_covariates]
        :param compute_loss: if True, compute and return the loss
        :param do_average: if True, average the loss over the minibatch
        :param eta_bn_prop: (float) a weight between 0 and 1 to interpolate between using and not using the final batchnorm layer
        :param var_scale: (float) a parameter which can be used to scale the variance of the random noise in the VAE
        :param l1_beta: np.array of prior variances for the topic weights
        :param l1_beta_c: np.array of prior variances on topic covariate deviations
        :param l1_beta_ci: np.array of prior variances on topic-covariate interactions
        :return: document representation; reconstruction; label probs; (loss, if requested)
        """

        # embed the word counts # 将词频转为嵌入
        en0_x = self.embeddings_x_layer(X) # [batch_size, emb_size]
        encoder_parts = [en0_x] # 多余代码 无用

        # append additional components to the encoder, if given
        # if self.n_prior_covars > 0:
        #     encoder_parts.append(PC)
        # if self.n_topic_covars > 0:
        #     encoder_parts.append(TC)
        # if self.n_labels > 0:
        #     encoder_parts.append(Y)

        if len(encoder_parts) > 1:
            en0 = torch.cat(encoder_parts, dim=1).to(self.device)
        else:
            en0 = en0_x
        
        

        ############################################################################################################################

        encoder_output = F.softplus(en0) # 维度不变 [batch_size, emb_size] 只是激活函数
        encoder_output_do = self.encoder_dropout_layer(encoder_output) # 维度不变 [batch_size, emb_size] 编码器层进行dropout

        

        ############################################################################################################################
        
        # 进入VAE阶段 compute the mean and variance of the document posteriors 计算文档先验的方差和均值
        posterior_mean = self.mean_layer(encoder_output_do) # 均值 [batch_size, emb_size] -> [batch_size, topic_size]
        posterior_logvar = self.logvar_layer(encoder_output_do) # 方差 维度不变 [batch_size, topic_size]

        posterior_mean_bn = self.mean_bn_layer(posterior_mean) # [batch_size, topic_size] 归一化 维度不变
        posterior_logvar_bn = self.logvar_bn_layer(posterior_logvar) # [batch_size, topic_size] 归一化 维度不变

        
        #posterior_mean_bn = posterior_mean
        #posterior_logvar_bn = posterior_logvar

        ############################################################################################################################

        posterior_var = posterior_logvar_bn.exp().to(self.device) # [batch_size, topic_size] 后验方差 维度不变
        

        # sample noise from a standard normal 采样噪声，用于VAE中
        eps = X.data.new().resize_as_(posterior_mean_bn.data).normal_().to(self.device) # [batch_size, topic_size]
        

        # compute the sampled latent representation
        z = posterior_mean_bn + posterior_var.sqrt() * eps * var_scale # [batch_size, topic_size] 得到噪声采样后的向量表示
        z_do = self.z_dropout_layer(z) # [batch_size, topic_size] 进行dropout操作
        

        # pass the document representations through a softmax 用softmax得到潜在表示
        theta = F.softmax(z_do, dim=1) # [batch_size, topic_size]
        

        ############################################################################################################################

        # transformer融合！
        if merge:
            theta_emb = self.topic2emb(theta)
            merge_theta = self.transformer_layer(struct_X, theta_emb)
        else:
            merge_theta = None
            

        ############################################################################################################################

        # combine latent representation with topics and background
        # beta layer here includes both the topic weights and the background term (as a bias)
        eta = self.beta_layer(theta) # [batch_size, vocab_size] 将潜在表示与之前计算的文档背景结合
        

        # add deviations for covariates (and interactions)
        # if self.n_topic_covars > 0:
        #     eta = eta + self.beta_c_layer(TC)
        #     if self.use_interactions:
        #         theta_rsh = theta.unsqueeze(2)
        #         tc_emb_rsh = TC.unsqueeze(1)
        #         covar_interactions = theta_rsh * tc_emb_rsh
        #         batch_size, _, _ = covar_interactions.shape
        #         eta += self.beta_ci_layer(covar_interactions.reshape((batch_size, self.n_topics * self.n_topic_covars)))

        # pass the unnormalized word probabilities through a batch norm layer
        eta_bn = self.eta_bn_layer(eta) # [batch_size, vocab_size] 批归一化 维度不变
        
        #eta_bn = eta

        ############################################################################################################################

        # compute X recon with and without batchnorm on eta, and take a convex combination of them
        X_recon_bn = F.softmax(eta_bn, dim=1) # [batch_size, vocab_size] 重建有批归一化的数据
        X_recon_no_bn = F.softmax(eta, dim=1) # [batch_size, vocab_size] 重建无批归一化的数据
        X_recon = eta_bn_prop * X_recon_bn + (1.0 - eta_bn_prop) * X_recon_no_bn # [batch_size, vocab_size]
        

        # predict labels
        Y_recon = None

        ############################################################################################################################

        if contrastive:
            pos_en0_x = self.embeddings_x_layer(pos_X)
            pos_encoder_parts = [pos_en0_x]

            neg_en0_x = self.embeddings_x_layer(neg_X)
            neg_encoder_parts = [neg_en0_x]

            if len(pos_encoder_parts) > 1:
                pos_en0 = torch.cat(pos_encoder_parts, dim=1).to(self.device)
            else:
                pos_en0 = pos_en0_x
    
            if len(neg_encoder_parts) > 1:
                neg_en0 = torch.cat(neg_encoder_parts, dim=1).to(self.device)
            else:
                neg_en0 = neg_en0_x

            pos_encoder_output = F.softplus(pos_en0) # 维度不变 [batch_size, emb_size] 只是激活函数
            pos_encoder_output_do = self.encoder_dropout_layer(pos_encoder_output) # 维度不变 [batch_size, emb_size] 编码器层进行dropout
    
            neg_encoder_output = F.softplus(neg_en0) # 维度不变 [batch_size, emb_size] 只是激活函数
            neg_encoder_output_do = self.encoder_dropout_layer(neg_encoder_output) # 维度不变 [batch_size, emb_size] 编码器层进行dropout

            pos_posterior_mean = self.mean_layer(pos_encoder_output_do) # 均值 [batch_size, emb_size] -> [batch_size, topic_size]
            pos_posterior_logvar = self.logvar_layer(pos_encoder_output_do) # 方差 维度不变 [batch_size, topic_size]
    
            pos_posterior_mean_bn = self.mean_bn_layer(pos_posterior_mean) # [batch_size, topic_size] 归一化 维度不变
            pos_posterior_logvar_bn = self.logvar_bn_layer(pos_posterior_logvar) # [batch_size, topic_size] 归一化 维度不变
    
            neg_posterior_mean = self.mean_layer(neg_encoder_output_do) # 均值 [batch_size, emb_size] -> [batch_size, topic_size]
            neg_posterior_logvar = self.logvar_layer(neg_encoder_output_do) # 方差 维度不变 [batch_size, topic_size]
    
            neg_posterior_mean_bn = self.mean_bn_layer(neg_posterior_mean) # [batch_size, topic_size] 归一化 维度不变
            neg_posterior_logvar_bn = self.logvar_bn_layer(neg_posterior_logvar) # [batch_size, topic_size] 归一化 维度不变

            pos_posterior_var = pos_posterior_logvar_bn.exp().to(self.device) # [batch_size, topic_size] 后验方差 维度不变
            neg_posterior_var = neg_posterior_logvar_bn.exp().to(self.device) # [batch_size, topic_size] 后验方差 维度不变
            pos_eps = pos_X.data.new().resize_as_(pos_posterior_mean_bn.data).normal_().to(self.device) # [batch_size, topic_size]
            neg_eps = neg_X.data.new().resize_as_(neg_posterior_mean_bn.data).normal_().to(self.device) # [batch_size, topic_size]
            pos_z = pos_posterior_mean_bn + pos_posterior_var.sqrt() * pos_eps * var_scale # [batch_size, topic_size] 得到噪声采样后的向量表示
            pos_z_do = self.z_dropout_layer(pos_z) # [batch_size, topic_size] 进行dropout操作
            neg_z = neg_posterior_mean_bn + neg_posterior_var.sqrt() * neg_eps * var_scale # [batch_size, topic_size] 得到噪声采样后的向量表示
            neg_z_do = self.z_dropout_layer(neg_z) # [batch_size, topic_size] 进行dropout操作
            pos_theta = F.softmax(pos_z_do, dim=1) # [batch_size, topic_size]
            neg_theta = F.softmax(neg_z_do, dim=1) # [batch_size, topic_size]

            pos_eta = self.beta_layer(pos_theta) # [batch_size, vocab_size] 将潜在表示与之前计算的文档背景结合
            neg_eta = self.beta_layer(neg_theta) # [batch_size, vocab_size] 将潜在表示与之前计算的文档背景结合
            pos_eta_bn = self.eta_bn_layer(pos_eta) # [batch_size, vocab_size] 批归一化 维度不变
            neg_eta_bn = self.eta_bn_layer(neg_eta) # [batch_size, vocab_size] 批归一化 维度不变
            pos_X_recon_bn = F.softmax(pos_eta_bn, dim=1) # [batch_size, vocab_size] 重建有批归一化的数据
            pos_X_recon_no_bn = F.softmax(pos_eta, dim=1) # [batch_size, vocab_size] 重建无批归一化的数据
            pos_X_recon = eta_bn_prop * pos_X_recon_bn + (1.0 - eta_bn_prop) * pos_X_recon_no_bn # [batch_size, vocab_size]
            neg_X_recon_bn = F.softmax(neg_eta_bn, dim=1) # [batch_size, vocab_size] 重建有批归一化的数据
            neg_X_recon_no_bn = F.softmax(neg_eta, dim=1) # [batch_size, vocab_size] 重建无批归一化的数据
            neg_X_recon = eta_bn_prop * neg_X_recon_bn + (1.0 - eta_bn_prop) * neg_X_recon_no_bn # [batch_size, vocab_size]

            if merge:
                pos_theta_emb = self.topic2emb(pos_theta)
                pos_merge_theta = self.transformer_layer(struct_X, pos_theta_emb)
                neg_theta_emb = self.topic2emb(neg_theta)
                neg_merge_theta = self.transformer_layer(struct_X, neg_theta_emb)
            else:
                pos_merge_theta = None
                neg_merge_theta = None
        
        else:
            pos_X_recon = None
            neg_X_recon = None
            pos_theta = None
            neg_theta = None
            pos_merge_theta = None
            neg_merge_theta = None

        ############################################################################################################################

        # if self.n_labels > 0:

        #     classifier_inputs = [theta]
        #     if self.classify_from_covars:
        #         if self.n_prior_covars > 0:
        #             classifier_inputs.append(PC)
        #         if self.n_topic_covars > 0:
        #             classifier_inputs.append(TC)

        #     if len(classifier_inputs) > 1:
        #         classifier_input = torch.cat(classifier_inputs, dim=1).to(self.device)
        #     else:
        #         classifier_input = theta

        #     if self.classifier_layers == 0:
        #         decoded_y = self.classifier_layer_0(classifier_input)
        #     elif self.classifier_layers == 1:
        #         cls0 = self.classifier_layer_0(classifier_input)
        #         cls0_sp = F.softplus(cls0)
        #         decoded_y = self.classifier_layer_1(cls0_sp)
        #     else:
        #         cls0 = self.classifier_layer_0(classifier_input)
        #         cls0_sp = F.softplus(cls0)
        #         cls1 = self.classifier_layer_1(cls0_sp)
        #         cls1_sp = F.softplus(cls1)
        #         decoded_y = self.classifier_layer_2(cls1_sp)
        #     Y_recon = F.softmax(decoded_y, dim=1)

        # compute the document prior if using prior covariates
        if self.n_prior_covars > 0:
            prior_mean = self.prior_covar_weights(PC)
            prior_logvar = self.prior_logvar.expand_as(posterior_logvar)
        else:
            prior_mean   = self.prior_mean.expand_as(posterior_mean)
            prior_logvar = self.prior_logvar.expand_as(posterior_logvar)

        if compute_loss: # 计算loss 返回theta, X_recon, Y_recon, loss.mean(), NL.mean(), KLD.mean()
            if merge:
                return merge_theta, X_recon, Y_recon, self._loss(contrastive, merge, X, Y, X_recon, pos_X_recon, neg_X_recon, theta, pos_theta, neg_theta, merge_theta, pos_merge_theta, neg_merge_theta, Y_recon, prior_mean, prior_logvar, posterior_mean_bn, posterior_logvar_bn, do_average, l1_beta, l1_beta_c, l1_beta_ci)
            else:
                return theta, X_recon, Y_recon, self._loss(contrastive, merge, X, Y, X_recon, pos_X_recon, neg_X_recon, theta, pos_theta, neg_theta, merge_theta, pos_merge_theta, neg_merge_theta, Y_recon, prior_mean, prior_logvar, posterior_mean_bn, posterior_logvar_bn, do_average, l1_beta, l1_beta_c, l1_beta_ci)
        
        else:
            if merge:
                return merge_theta, X_recon, Y_recon
            else:
                return theta, X_recon, Y_recon
            

    def _loss(self, contrastive, merge, X, Y, X_recon, pos_X_recon, neg_X_recon, theta, pos_theta, neg_theta, merge_theta, pos_merge_theta, neg_merge_theta, Y_recon, prior_mean, prior_logvar, posterior_mean, posterior_logvar, do_average=True, l1_beta=None, l1_beta_c=None, l1_beta_ci=None):

        # compute reconstruction loss
        NL = -(X * (X_recon+1e-10).log()).sum(1) # 词频矩阵和重构矩阵求loss
        # compute label loss
        # if self.n_labels > 0:
        #     NL += -(Y * (Y_recon+1e-10).log()).sum(1)

        # compute KLD
        prior_var = prior_logvar.exp()
        posterior_var = posterior_logvar.exp()
        var_division    = posterior_var / prior_var
        diff            = posterior_mean - prior_mean
        diff_term       = diff * diff / prior_var
        logvar_division = prior_logvar - posterior_logvar

        # put KLD together
        KLD = 0.5 * ((var_division + diff_term + logvar_division).sum(1) - self.n_topics)

        if contrastive:
            # 定义损失函数
            criterion = cl.InfoNCELoss(temperature=0.07)
            
            contrastive_loss = criterion(theta, pos_theta, neg_theta)
    
            # combine
            loss = (NL + KLD + contrastive_loss)
        else:
            loss = (NL + KLD)

        

        # # add regularization on prior
        # if self.l2_prior_reg > 0 and self.n_prior_covars > 0:
        #     loss += self.l2_prior_reg * torch.pow(self.prior_covar_weights.weight, 2).sum()

        # # add regularization on topic and topic covariate weights
        # if self.l1_beta_reg > 0 and l1_beta is not None:
        #     l1_strengths_beta = torch.from_numpy(l1_beta).to(self.device)
        #     beta_weights_sq = torch.pow(self.beta_layer.weight, 2)
        #     loss += self.l1_beta_reg * (l1_strengths_beta * beta_weights_sq).sum()

        # if self.n_topic_covars > 0 and l1_beta_c is not None and self.l1_beta_c_reg > 0:
        #     l1_strengths_beta_c = torch.from_numpy(l1_beta_c).to(self.device)
        #     beta_c_weights_sq = torch.pow(self.beta_c_layer.weight, 2)
        #     loss += self.l1_beta_c_reg * (l1_strengths_beta_c * beta_c_weights_sq).sum()

        # if self.n_topic_covars > 0 and self.use_interactions and l1_beta_c is not None and self.l1_beta_ci_reg > 0:
        #     l1_strengths_beta_ci = torch.from_numpy(l1_beta_ci).to(self.device)
        #     beta_ci_weights_sq = torch.pow(self.beta_ci_layer.weight, 2)
        #     loss += self.l1_beta_ci_reg * (l1_strengths_beta_ci * beta_ci_weights_sq).sum()

        # average losses if desired
        if do_average:
            if contrastive:
                return loss.mean(), NL.mean(), KLD.mean(), contrastive_loss.mean()
            else:
                return loss.mean(), NL.mean(), KLD.mean()
        else:
            if contrastive:
                return loss, NL, KLD, contrastive_loss
            else:
                return loss, NL, KLD
            

    def predict_from_theta(self, theta, PC, TC):
        # Predict labels from a distribution over topics
        Y_recon = None
        if self.n_labels > 0:

            classifier_inputs = [theta]
            if self.classify_from_covars:
                if self.n_prior_covars > 0:
                    classifier_inputs.append(PC)
                if self.n_topic_covars > 0:
                    classifier_inputs.append(TC)
            if len(classifier_inputs) > 1:
                classifier_input = torch.cat(classifier_inputs, dim=1).to(self.device)
            else:
                classifier_input = theta.to(self.device)

            if self.classifier_layers == 0:
                decoded_y = self.classifier_layer_0(classifier_input)
            elif self.classifier_layers == 1:
                cls0 = self.classifier_layer_0(classifier_input)
                cls0_sp = F.softplus(cls0)
                decoded_y = self.classifier_layer_1(cls0_sp)
            else:
                cls0 = self.classifier_layer_0(classifier_input)
                cls0_sp = F.softplus(cls0)
                cls1 = self.classifier_layer_1(cls0_sp)
                cls1_sp = F.softplus(cls1)
                decoded_y = self.classifier_layer_1(cls1_sp)
            Y_recon = F.softmax(decoded_y, dim=1)

        return Y_recon