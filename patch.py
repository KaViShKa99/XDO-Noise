import torch
import torch.nn.functional as F
import torchvision
import time
from tqdm import tqdm
import numpy as np
import argparse
from utils import get_log_writer, scale_theta, scale_pattern
from models import load_generator, load_model_vggface
from data import load_facedata_unnormalized, normalize_vggface, load_facedata_unnormalized_facenet, load_imagenet_unnormalize, normalize_imagenet
import torchvision.models as models
from torchvision import utils as vutils
import math
from PIL import Image
import torchvision.transforms as transforms
from utils import get_attack_model
from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
from NES_GMM import PEPG
from load_face_models.load_face_model import load_reg_model, load_cls_model


import random
import os
# from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization, training

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
     os.environ['PYTHONHASHSEED'] = str(seed)

setup_seed(2023)


def get_args():
    # input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='test')
    parser.add_argument('--patch_size', type=float, default=0.25)
    # parser.add_argument('--img_size', type=int, default=112)

    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--beta', type=int, default=3000)
    parser.add_argument('--dataset', type=str, default='vggface')
    parser.add_argument('--data_path', type=str, default='./data/lfw-align-128_cropped')
    parser.add_argument('--sticker_path', type=str, default='./img/sticker_test.png')
    parser.add_argument('--sticker_id', type=int, default=0)

    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr_gen', type=float, default=0.05)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--device', type=str, default='cuda')

    parser.add_argument('--test_freq', type=int, default=10)
    parser.add_argument('--balck_attack_freq', type=int, default=30)

    parser.add_argument('--num_k', type=int, default=1)
    parser.add_argument('--lr_step', type=int, default=150)
    parser.add_argument('--task', type=str, default='cls', choices=['cls', 'reg'])
    parser.add_argument('--lamba', type=float, default=0.0001)
    parser.add_argument('--MC_sample', type=int, default=10)

    #NES
    parser.add_argument('--NES_lr', type=float, default=100)
    parser.add_argument('--NES_sigma', type=float, default=100)

    # get victim model
    parser.add_argument('--face_img', type=str, default='face', choices=['face', 'img'])
    # parser.add_argument('--backbone', type=str, default='Res50_IR', choices=['Facenet', 'MobileFace', 'Res50_IR', 'SERes50_IR', 'Res100_IR', 'SERes100_IR','Attention_56', 'Attention_92'])
    parser.add_argument('--model_name', type=str, default='facenet', choices=['facenet', 'arcface', 'cosface', 'sphereface', 'mobilefacenet','arcface_50', 'cosface_50', 'sphereface_50', 'multimargin_50'])
    parser.add_argument('--black_optim_method', type=str, default='random', choices=['random', 'BO', 'NES', 'No'])
    parser.add_argument('--only_black_optim', action='store_true', default=False)

    # generator
    parser.add_argument('--mp_arc', type=str, default='res', choices=['res', 'res-ir'])
    parser.add_argument('--G_path', type=str, default='./model_zoo/ArcFace_Res50_IR.ckpt')
    parser.add_argument('--mp_blocks', type=int, default=6)
    args = parser.parse_args()
    return args

def del_tensor_ele(arr,index):
    arr1 = arr[0:index]
    arr2 = arr[index+1:]
    return torch.cat((arr1,arr2),dim=0)

def CWLoss(logistic, label, k=0.3):
    t_value = logistic[int(label)]
    wo_label_logistic = del_tensor_ele(logistic, label)
    i_value = torch.max(wo_label_logistic)

    return torch.max(i_value-t_value, -k)



def entropy_loss(aff_theta, var, rand_noise, f):
    args = get_args()
    mu = aff_theta
    sigma = var
    r = rand_noise
    omiga = torch.tensor(1/args.num_k).to(args.device)

    Entropy = torch.zeros([aff_theta.shape[0], 3]).to(args.device)

    for i in range(3):
        for j in range(aff_theta.shape[0]):  
            k = int(f[j])
            inside = 1-torch.pow(torch.tanh(mu[j, 3*k+i] + sigma[j, 3*k+i] * r[j, 3*k+i]), 2) + 1e-8
            neg_logp = -torch.log(omiga+1e-8) + torch.log(sigma[j, 3*k+i]+1e-8) + 1/2*torch.pow(r[j, 3*k+i], 2) + torch.log(inside)
            Entropy[j, i] += neg_logp

    Entropy = torch.mean(Entropy)

    return Entropy


def fixed_image_standardization(image_tensor):
    processed_tensor = (image_tensor - 127.5) / 128.0
    return processed_tensor

def transorm_to_tensor(img_tensor, img_size):
    args = get_args()
    trans = transforms.Compose([
                                transforms.Resize([int(args.patch_size*img_size[0]), int(args.patch_size*img_size[1])]),
                                np.float32,
                                transforms.ToTensor(),
                                fixed_image_standardization
                                ])

    return trans(img_tensor)


def transorm_to_tensor_mask(img_tensor, img_size):
    args = get_args()
    
    trans = transforms.Compose([
                                transforms.Resize([int(args.patch_size*img_size[0]), int(args.patch_size*img_size[1])]),
                                np.float32,
                                transforms.ToTensor()])
    return trans(img_tensor)



def load_face_bank(dataloader, device, model):
    args = get_args()
    print('------build face reg bank-------')
    if args.model_name != 'mobilefacenet':
        e_l = 512
    else:
        e_l = 128
    clean_embedding = torch.zeros((13242, e_l))
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm(dataloader)):
            inputs, targets = inputs.to(device), targets.to(device)
            embedding_batch = model(inputs)
            for i in range(embedding_batch.shape[0]):
                clean_embedding[int(targets[i]),:] = embedding_batch[i]

    return clean_embedding.to(device)

def load_face_bank_black(dataloader, device, model, black_model_name, trans):
    args = get_args()
    print('------build face reg bank-------')
    if black_model_name != 'mobilefacenet':
        e_l = 512
    else:
        e_l = 128
    clean_embedding = torch.zeros((13242, e_l))
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm(dataloader)):
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = trans(inputs)
            embedding_batch = model(inputs)
            for i in range(embedding_batch.shape[0]):
                clean_embedding[int(targets[i]),:] = embedding_batch[i]

    return clean_embedding.to(device)

def save_adv_img(img_tnesor, batch_idx):
    # face_img = img_tnesor.view(3, img_tnesor.shape[-1]*int(math.sqrt(args.batch_size)), -1)
    img_tnesor = img_tnesor
    vutils.save_image(img_tnesor, f'./result/visual_adv_face-batch{batch_idx}.jpg', nrow=5, normalize=True)

def load_sticker(bs, img_size):
    args = get_args()
    path = args.sticker_path
    sticker_img = Image.open(path).convert('RGB')
    sticker_img = transorm_to_tensor(sticker_img, img_size).unsqueeze(0).repeat(bs, 1, 1, 1)
    
    sticker_mask = Image.open(path)
    sticker_mask = transorm_to_tensor_mask(sticker_mask, img_size).unsqueeze(0).repeat(bs, 1, 1, 1)
    sticker_mask = sticker_mask[:,3,:,:].unsqueeze(1)

    zero = torch.zeros_like(sticker_mask)
    one = torch.ones_like(sticker_mask)
    sticker_mask = torch.where(sticker_mask>0, one, zero)

    return sticker_img, sticker_mask

def move_m_p(aff_theta, device, img_size, alpha=1):
    args = get_args()

    bs = aff_theta.size()[0]
    sticker, alpha = load_sticker(bs, img_size)

    image_with_patch = torch.zeros(bs, 3, img_size[0], img_size[1], device=device)
    mask_with_patch = torch.zeros(bs, 1, img_size[0], img_size[1], device=device)
    start_w = (int(img_size[0]/2)) - sticker.size()[2] // 2
    end_w = start_w + sticker.size()[2]

    start_h = (int(img_size[1]/2)) - sticker.size()[3] // 2
    end_h = start_h + sticker.size()[3]

    image_with_patch[:, :, start_w:end_w, start_h:end_h] = sticker
    mask_with_patch[:, :, start_w:end_w, start_h:end_h] = alpha

    # ------------- ovo
    rot_theta = torch.tensor([[1.0, 0.0], [0.0, 1.0]]).unsqueeze(0).to(device).repeat(bs, 1, 1)
    theta_batch = torch.cat((rot_theta, aff_theta[:,:2].unsqueeze(2)), 2)
    
    r = aff_theta[:,2]*torch.pi
    theta_batch[:, 0, 0] = torch.cos(r)
    theta_batch[:, 0, 1] = torch.sin(-r)
    theta_batch[:, 1, 0] = torch.sin(r)
    theta_batch[:, 1, 1] = torch.cos(r)
    # ------------- ovo   
    
    grid = F.affine_grid(theta_batch, image_with_patch.size(), align_corners=True)
    sticker = F.grid_sample(image_with_patch, grid, align_corners=True)
    mask_s = F.grid_sample(mask_with_patch, grid, align_corners=True)
    return mask_s, sticker


def perturb_image(inputs, mp_generator, devide_theta, device, img_size, alpha=1, p_scale=10000):
    args = get_args()
    aff_theta, var = mp_generator(inputs)
    var = F.softplus(var)
    rand_noise = torch.randn_like(aff_theta, requires_grad=True)

    # ------------- ovo   
    f = torch.multinomial(1/(aff_theta.shape[1]/3)*torch.ones([aff_theta.shape[0], int(aff_theta.shape[1]/3)]), 1, replacement=True).to(device)
    aff_theta_x = aff_theta.gather(1, f*3) + rand_noise.gather(1, f*3)*var.gather(1, f*3)
    aff_theta_y = aff_theta.gather(1, f*3+1) + rand_noise.gather(1, f*3+1)*var.gather(1, f*3+1)
    aff_theta_r = aff_theta.gather(1, f*3+2) + rand_noise.gather(1, f*3+2)*var.gather(1, f*3+2)

    param = torch.cat((aff_theta_x, aff_theta_y, aff_theta_r), dim=1)
    # ------------- ovo
    param = scale_theta(param, devide_theta)
    # pattern_s = scale_pattern(pattern_generated, p_scale=p_scale)
    mask_s, pattern_s = move_m_p(param, device, img_size, alpha=alpha)
    inputs = inputs * (1 - mask_s) + pattern_s * mask_s
    inputs = inputs.clamp(-1, 1)
    # save_adv_img(inputs, 0)

    return inputs, aff_theta, var, rand_noise, f


def train_gen_batch(inputs, targets, model, mp_generator, optimizer_gen, criterion,
                    adv_loss_l_gen, entropy_loss_l_gen, devide_theta, normalize_func, device, task, clean_embedding, img_size, alpha=1, p_scale=10000):
    args = get_args()
    
    mp_generator.train() 
    model.eval()
    attack_success = 0
    # print(model)
    inputs, targets = inputs.to(device), targets.to(device)
    # print(targets)
    optimizer_gen.zero_grad()
    for i in range(args.MC_sample):
        adv_face, aff_theta, var, rand_noise, f = perturb_image(inputs, mp_generator, devide_theta, device, img_size, alpha=alpha, p_scale=p_scale)
        outputs = model(adv_face)
        # outputs = model.fc(outputs)
        # print('outputs:', outputs.shape)
        if task == 'cls':
            adv_loss = -criterion(outputs, targets)
            # print('loss:', loss.item())
        elif task == 'reg':
            embedding = model(inputs)
            loss_all = criterion(outputs, embedding)
            adv_loss = torch.mean(loss_all)
        
        loss_entropy = entropy_loss(aff_theta, var, rand_noise, f)
        loss = adv_loss - args.lamba * loss_entropy
        loss.backward(retain_graph=True if i != (args.MC_sample-1) else False)
        
    optimizer_gen.step()
    adv_loss_l_gen.append(adv_loss.item())
    entropy_loss_l_gen.append(loss_entropy.item())
    # save_adv_img(adv_face, 0)
    if task == 'cls':
        _, predicted = outputs.max(1)
        attack_success = (~predicted.eq(targets)).sum().item()
    elif task == 'reg':
        with torch.no_grad():
            for s in range(len(targets)):
                adv_embedding = outputs[s,:].repeat(13242, 1)
                id = torch.argmax(criterion(adv_embedding, clean_embedding))
                attack_success += 1 if id != targets[s] else 0

    return attack_success, targets.size(0), adv_face


def test_gen_batch(inputs, targets, model, mp_generator,
                   optimizer_gen, criterion, devide_theta, normalize_func, device, task, clean_embedding, img_size, alpha=1, p_scale=10000):
    correct = 0
    total = 0
    correct_m = np.zeros(targets.size(0))
    mp_generator.eval()
    model.eval()
    with torch.no_grad():
        for i in range(10):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer_gen.zero_grad()
            adv_face, aff_theta, var, rand_noise, f = perturb_image(inputs, mp_generator, devide_theta, device, img_size, alpha=alpha, p_scale=p_scale)
            outputs = model(adv_face)

            if task == 'reg':
                for s in range(len(targets)):
                    adv_embedding = outputs[s,:].repeat(13242, 1)
                    id = torch.argmax(criterion(adv_embedding, clean_embedding))
                    correct += 1 if id != targets[s] else 0
                    correct_m[s] += 1 if id != targets[s] else 0

            elif task == 'cls':
                _, predicted = outputs.max(1)
                correct += (~predicted.eq(targets)).sum().item()

                for j in range(targets.size(0)):
                    correct_m[j] += 1 if ~predicted[j].eq(targets[j]) else 0
            
            
            total += targets.size(0)
    
    soft_correct = np.sum(correct_m>0)
    return correct, total, adv_face, soft_correct

def test_clean(inputs, targets, model, mp_generator,
                   optimizer_gen, criterion, devide_theta, normalize_func, device, task, clean_embedding, alpha=1, p_scale=10000):
    correct = 0
    total = 0
    mp_generator.eval()
    model.eval()
    with torch.no_grad():
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer_gen.zero_grad()
        outputs = model(inputs)
        if task == 'reg':
            for s in range(len(targets)):
                adv_embedding = outputs[s,:].repeat(13242, 1)
                id = torch.argmax(criterion(adv_embedding, clean_embedding))
                correct += 1 if id == targets[s] else 0

        elif task == 'cls':
            _, predicted = outputs.max(1)
            correct += (predicted.eq(targets)).sum().item()
            
        total += targets.size(0)
    
    return correct, total

############################## stage2-black-box-optim #################################################

# ----------------------------------------- NES-distribution-transfer ------------------------------------------------------
def NES_evaluate(black_model, param, face_img, label, criterion, device, black_img_size, clean_embedding, attack_type='dod'):
    success = False
    batch_query = len(param)
    args = get_args()
    mask_s, pattern_s = move_m_p(param, device, black_img_size, alpha=args.alpha)
    adv_faces = face_img * (1 - mask_s) + pattern_s * mask_s
    adv_faces = adv_faces.clamp(-1, 1)
    outputs = black_model(adv_faces)
    _, predicted = torch.max(outputs, dim=1)

    if args.task == 'cls':
        loss = criterion(outputs, label.repeat(outputs.shape[0]))
    elif args.task == 'reg':
        embedding = black_model(face_img)
        loss = -criterion(outputs, embedding)
    
    for t in range(len(predicted.cpu().numpy())):
        if args.task == 'cls': 
            if predicted.cpu().numpy()[t] != label.cpu().numpy():
                success = True
                batch_query = t
        elif args.task == 'reg':
            adv_embedding = outputs[t,:].repeat(13242, 1)
            id = torch.argmax(criterion(adv_embedding, clean_embedding))
            if id.cpu().numpy() != label.cpu().numpy():
                success = True
                batch_query = t

    return loss.cpu().numpy(), predicted.cpu().numpy(), success, batch_query


def NES_Optim(black_model, black_model_name, Generator, dataloader, device, black_img_size, trans_to_black_input):
    
    args = get_args()
    attack_success = 0
    total_query = 0
    toatal = 0
    if args.task == 'cls':
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        clean_embedding = None
    elif args.task == 'reg':
        criterion = torch.nn.CosineSimilarity()
        clean_embedding = load_face_bank_black(dataloader, device, black_model, black_model_name, trans_to_black_input)

    MAX_ITERATION = 100
    POPSIZE = 21
    NUM_PARAMS = 3
    NUM_K = args.num_k

    for batch_idx, (inputs, targets) in enumerate(tqdm(dataloader)):
        for i in range(targets.size(0)):
            query_num = 0

            img, target = inputs[i,:,:,:].to(device).unsqueeze(0), targets[i].to(device).unsqueeze(0)
            with torch.no_grad():
                aff_theta, var = Generator(img)
            var = F.softplus(var)
            aff_theta, var = aff_theta.cpu().numpy(), var.cpu().numpy()

            aff_theta = aff_theta.reshape((args.num_k, NUM_PARAMS))
            var = var.reshape((args.num_k, NUM_PARAMS))
            omega = (1/args.num_k)*np.ones_like(aff_theta)
            
            img = trans_to_black_input(img)

            solver = PEPG(num_params=NUM_PARAMS,  # number of model parameters
                  num_k = NUM_K,
                  sigma_init=args.NES_sigma,  # initial standard deviation
                  sigma_update=True,  # 不大幅更新sigma
                  learning_rate=args.NES_lr,  # learning rate for standard deviation
                  learning_rate_decay=0.99, # don't anneal the learning rate
                  learning_rate_limit=0,
                  popsize=POPSIZE,  # population size
                  average_baseline=False,  # set baseline to average of batch
                  weight_decay=0.00,  # weight decay coefficient
                  rank_fitness=True,  # use rank rather than fitness numbers
                  forget_best=False,
                  mu_lambda=0,
                  sigma_lambda=0,
                  omiga_lamba=0,
                  random_begin=True,
                  omiga_alpha=0.02,
                  update_omiga=False,
                  start_mu = aff_theta,
                  start_sigma = var,
                  start_omega = omega
                  )

            history = []
            fitness_origin = []
            history_best_solution = []
            for j in range(MAX_ITERATION):
                solutions = solver.ask()
                mu_entropy_grad, sigma_entropy_grad, omiga_entropy_grad = solver.comput_entropy()

                param = torch.FloatTensor(solutions).to(device)
                param = scale_theta(param, args.beta)

                # fitness_list = np.zeros(solver.popsize+solver.num_k-1)
                # pred_y_list = np.zeros(solver.popsize+solver.num_k-1)
                
                with torch.no_grad():
                    fitness_list, pred_y_list, success, batch_query = NES_evaluate(black_model, param, img, target, criterion, device, black_img_size, clean_embedding, attack_type='dod')
                
                query_num += batch_query
                
                if success:
                    attack_success += 1
                    break

                solver.tell(fitness_list, mu_entropy_grad, sigma_entropy_grad, omiga_entropy_grad)
                result = solver.result()  # first element is the best solution, second element is the best fitness

                fitness_origin.append(np.max(fitness_list))
                history.append(result[1])
                average_fitness = np.mean(fitness_list)
                max_idx = np.argmax(fitness_list)
                history_best_solution.append(solutions[max_idx])
                # print("fitness at iteration\n", (j + 1), max(fitness_origin))
                print("average fitness at iteration\n", (j + 1), average_fitness)

            if success:
                total_query += query_num
            toatal += 1
    
    return total_query/attack_success, attack_success/toatal

# ----------------------------------------- BO-location-optim ------------------------------------------------------
class black_box_function:
    def __init__(self, device, args, black_model, criterion, inputs, targets, black_img_size, clean_embedding):
        self.device = device
        self.args = args
        self.black_model = black_model
        self.criterion = criterion
        self.inputs = inputs
        self.targets = targets
        self.query = 0
        self.success = False
        self.black_img_size = black_img_size
        self.clean_embedding = clean_embedding
    
    def forward(self, x, y, r):
        param = torch.FloatTensor([[x,y,r]]).to(self.device)
        param = scale_theta(param, self.args.beta)
        mask_s, pattern_s = move_m_p(param, self.device, self.black_img_size, alpha=self.args.alpha)
        adv_faces = self.inputs * (1 - mask_s) + pattern_s * mask_s
        adv_faces = adv_faces.clamp(-1, 1)

        # 传入 black_model
        outputs = self.black_model(adv_faces)
        _, predicted = outputs.max(1)

        if self.args.task == 'cls':
            self.success = True if ~predicted.eq(self.targets) else False
        elif self.args.task == 'reg':
            adv_embedding = outputs.repeat(13242, 1)
            id = torch.argmax(self.criterion(adv_embedding, self.clean_embedding))
            self.success = True if id != self.targets else False

        if not self.success:
            self.query += 1

        if self.args.task == 'cls':
            loss = self.criterion(outputs, self.targets)
        elif self.args.task == 'reg':
            embedding = self.black_model(self.inputs)
            loss_all = self.criterion(outputs, embedding)
            loss = -torch.mean(loss_all)
        
        return loss.cpu().numpy()


def Bayesian_Optim(black_model, black_model_name, Generator, dataloader, device, black_img_size, trans_to_black_input):
    args = get_args()
    attack_success = 0
    total_query = 0
    toatal = 0
    if args.task == 'cls':
        criterion = torch.nn.CrossEntropyLoss()
        clean_embedding = None
    elif args.task == 'reg':
        criterion = torch.nn.CosineSimilarity()
        clean_embedding = load_face_bank_black(dataloader, device, black_model, black_model_name, trans_to_black_input)
    ###############################################################################
    for batch_idx, (inputs, targets) in enumerate(tqdm(dataloader)):
        for i in range(targets.size(0)):
            if toatal < 128:
                img, target = inputs[i,:,:,:].to(device).unsqueeze(0), targets[i].to(device).unsqueeze(0)
                aff_theta, var = Generator(img)
                var = F.softplus(var)
                aff_theta, var = aff_theta.cpu().numpy(), var.cpu().numpy()
                img = trans_to_black_input(img)

                # Bounded region of parameter space
                # 在5σ范围内
                pbounds = {'x': (aff_theta[0,0]-10*var[0,0], aff_theta[0,0]+10*var[0,0]), 
                        'y': (aff_theta[0,1]-10*var[0,1], aff_theta[0,1]+10*var[0,1]), 
                        'r': (aff_theta[0,2]-10*var[0,2], aff_theta[0,2]+10*var[0,2])}

                black_box_f = black_box_function(device, args, black_model, criterion, img, target, black_img_size, clean_embedding)

                optimizer = BayesianOptimization(
                            f=black_box_f.forward,
                            pbounds=pbounds,
                            random_state=1,
                            allow_duplicate_points=True
                            )

                # utility = UtilityFunction(kind="ucb", kappa=2.5, xi=0.0)
                # for _ in range(5):
                #     next_point = optimizer.suggest(utility)
                #     target = black_box_function.forward(next_point['x'], next_point['y'], next_point['r'])
                #     optimizer.register(params=next_point, target=target)
                    
                #     print(target, next_point)
                #     print(optimizer.max)

                optimizer.maximize(init_points=100, n_iter=200, black_box_f=black_box_f)
                # print(optimizer.max)
                if black_box_f.success:
                    attack_success += 1
                    total_query += black_box_f.query+1
                
                print("its:", toatal)
                print("attack_success", attack_success)
                print("total_query", total_query)

                toatal += 1
    
    return total_query/attack_success, attack_success/toatal

# ----------------------------------------- Rabdom-location-optim ------------------------------------------------------
def Random_Search(args, black_model, black_model_name, Generator, dataloader, device, black_img_size, trans_to_black_input, upper_query=1000):
    
    attack_success = 0
    total_query = 0
    toatal = 0
    if args.task == 'cls':
        criterion = torch.nn.CrossEntropyLoss()
        clean_embedding = None
    elif args.task == 'reg':
        criterion = torch.nn.CosineSimilarity()
        clean_embedding = load_face_bank_black(dataloader, device, black_model, black_model_name, trans_to_black_input)

    for batch_idx, (inputs, targets) in enumerate(tqdm(dataloader)):
        inputs, targets = inputs.to(device), targets.to(device)

        query_map = np.zeros(targets.size(0))
        success_map = np.zeros(targets.size(0))

        aff_theta, var = Generator(inputs)
        var = F.softplus(var)

        inputs = trans_to_black_input(inputs)
        for i in range(upper_query):
            # 获取初始攻击参数

            rand_noise = torch.randn_like(aff_theta, requires_grad=True)
            f = torch.multinomial(1/(aff_theta.shape[1]/3)*torch.ones([aff_theta.shape[0], int(aff_theta.shape[1]/3)]), 1, replacement=True).to(device)
            aff_theta_x = aff_theta.gather(1, f*3) + rand_noise.gather(1, f*3)*var.gather(1, f*3)*3
            aff_theta_y = aff_theta.gather(1, f*3+1) + rand_noise.gather(1, f*3+1)*var.gather(1, f*3+1)*3
            aff_theta_r = aff_theta.gather(1, f*3+2) + rand_noise.gather(1, f*3+2)*var.gather(1, f*3+2)*3

            # aff_theta_x = 1000*torch.randn_like(aff_theta_x, requires_grad=True)
            # aff_theta_y = 1000*torch.randn_like(aff_theta_y, requires_grad=True)
            # aff_theta_r = 1000*torch.randn_like(aff_theta_r, requires_grad=True)

            param = torch.cat((aff_theta_x, aff_theta_y, aff_theta_r), dim=1)
            param = scale_theta(param, args.beta)

            # 生成Adv_face
            mask_s, pattern_s = move_m_p(param, device, black_img_size, alpha=args.alpha)
            adv_faces = inputs * (1 - mask_s) + pattern_s * mask_s
            adv_faces = adv_faces.clamp(-1, 1)

            # 传入 black_model
            outputs = black_model(adv_faces)
            
            if args.task == 'reg':
                for s in range(len(targets)):
                    adv_embedding = outputs[s,:].repeat(13242, 1)
                    id = torch.argmax(criterion(adv_embedding, clean_embedding))
                    success_map[s] += 1 if id != targets[s] else 0
                    query_map[s] += 1 if success_map[s] == 0 else 0

            elif args.task == 'cls':
                _, predicted = outputs.max(1)
                for j in range(targets.size(0)):
                    success_map[j] += 1 if ~predicted[j].eq(targets[j]) else 0
                    query_map[j] += 1 if success_map[j] == 0 else 0
        
        toatal += targets.size(0)
        for i in range(len(query_map)):
            total_query += query_map[i] if query_map[i]<upper_query else 0
        attack_success += np.sum(success_map>0)
    
    average_query = total_query/attack_success
    average_success = attack_success/toatal

    return average_query, average_success

def balckbox_attack(writer, epoch, mp_generator, dataloader, method, id_num):
    args = get_args()
    if args.task == 'cls':
        all_models = ['facenet', 'arcface', 'cosface']
    else:
        all_models = ['facenet', 'arcface_50', 'arcface', 'mobilefacenet']
    # all_models = ['facenet']
    # all_models = ['MultiMargin']
    # all_models.remove(args.metric)
    asr_dict = []
    for model_name in all_models:
        # ---------------------------------------------------------------------------------------
        if args.task == 'cls':
            balck_model = load_cls_model(model_name=model_name, device=args.device, id_num=id_num)
        elif args.task == 'reg':
            balck_model = load_reg_model(model_name=model_name, device=args.device)
        
        if model_name == 'facenet':
            black_img_size = [160, 160]
        elif model_name == 'mobilefacenet':
            black_img_size = [112, 96]
        else:
            black_img_size = [112, 112]
        
        trans_to_black_input = transforms.Resize(black_img_size)
        # ---------------------------------------------------------------------------------------

        print(f'attack balck model:{model_name}')
        # 调用优化函数
        if method == 'random':
            with torch.no_grad():
                average_query, average_success = Random_Search(args, balck_model, model_name, mp_generator, dataloader, args.device, black_img_size, trans_to_black_input)
        elif method == 'BO':
            with torch.no_grad():
                average_query, average_success = Bayesian_Optim(balck_model, model_name, mp_generator, dataloader, args.device, black_img_size, trans_to_black_input)
        elif method == 'NES':
            average_query, average_success = NES_Optim(balck_model, model_name, mp_generator, dataloader, args.device, black_img_size, trans_to_black_input)
        asr_dict.append(model_name)
        asr_dict.append(average_query)
        asr_dict.append(average_success)
        
        print('+++++++++++++++++++++++++++++++++++++++++++')
        print('method:', args.black_optim_method)
        print('average_query:', average_query)
        print('average_success:', average_success)

        # writer.add_scalar(f'balck_eval/{model_name}_AttackAcc', average_success, epoch)
        # writer.add_scalar(f'balck_eval/{model_name}_Query', average_query, epoch)
    with open(f'./result/BOresult-{model_name}.txt','w') as fp:
        [fp.write(str(item)+'\n') for  item in asr_dict]
        fp.close()

############################## stage2-black-box-optim-over ################################################# 

def DOPatch(dataloader, dataloader_val, dataloader_all, model, mp_generator, optimizer_gen, scheduler, criterion,
          epochs, devide_theta, alpha, task, normalize_func, writer, device, img_size, id_num):
    args = get_args()
        
    if task == 'reg':
        clean_embedding = load_face_bank(dataloader, device, model)
    else:
        clean_embedding = None

    # clean_test
    total_clean = 0
    correct_clean = 0
    for batch_idx, (inputs, targets) in enumerate(tqdm(dataloader)):
        correct_batch, total_batch = test_clean(inputs, targets, model,
                                                                            mp_generator,
                                                                            optimizer_gen, criterion, devide_theta, normalize_func,
                                                                            device, task, clean_embedding, alpha=alpha, p_scale=10000)

        correct_clean += correct_batch
        total_clean += total_batch
        # testing log
    asr = correct_clean / total_clean
    print("clean acc:", asr)


    if args.only_black_optim:
        if args.black_optim_method != 'No':
                balckbox_attack(writer, 0, mp_generator.eval(), dataloader, args.black_optim_method, id_num)
        else:
            pass

    else:
        for epoch in range(epochs):
            start_time = time.time()
            print('epoch: {}'.format(epoch))

            if epoch % args.balck_attack_freq == 0:
                if args.black_optim_method != 'No':
                    balckbox_attack(writer, epoch, mp_generator.eval(), dataloader, args.black_optim_method, id_num)
                else:
                    pass

            # testing
            if epoch % args.test_freq == 0:
                correct_gen2 = 0
                total_gen2 = 0
                soft_correct = 0
                for batch_idx, (inputs, targets) in enumerate(tqdm(dataloader)):
                    correct_batch, total_batch, final_ims_gen, soft_correct_batch = test_gen_batch(inputs, targets, model,
                                                                            mp_generator,
                                                                            optimizer_gen, criterion, devide_theta, normalize_func,
                                                                            device, task, clean_embedding, img_size, alpha=alpha, p_scale=10000)
                    correct_gen2 += correct_batch
                    total_gen2 += total_batch
                    soft_correct += soft_correct_batch
                # testing log
                asr = correct_gen2 / total_gen2
                soft_asr = (soft_correct/total_gen2)*10
                print("attack acc:", asr, soft_asr)
                writer.add_scalar('test_gen/asr', asr, epoch)
                writer.add_scalar('test_gen/asr_soft', soft_asr, epoch)
                final_ims_gen = torchvision.utils.make_grid(final_ims_gen, normalize=True)
                writer.add_image('final_im_test/{}'.format(epoch), final_ims_gen, epoch)
            
            # training
            adv_loss_l_gen = []
            entropy_loss_l_gen = []
            correct_gen = 0
            total_gen = 0
            for batch_idx, (inputs, targets) in enumerate(tqdm(dataloader)):
                correct_batch, total_batch, final_ims_gen = train_gen_batch(inputs, targets, model,
                                                                            mp_generator,
                                                                            optimizer_gen, criterion,
                                                                            adv_loss_l_gen, entropy_loss_l_gen, devide_theta, normalize_func,
                                                                            device, task, clean_embedding, img_size, alpha=alpha, p_scale=10000)
                correct_gen += correct_batch
                total_gen += total_batch
            # training log
            adv_loss = np.array(adv_loss_l_gen).mean()
            entropy_loss = np.array(entropy_loss_l_gen).mean()

            asr = correct_gen / total_gen
            writer.add_scalar('train_gen/adv_loss', adv_loss, epoch)
            writer.add_scalar('train_gen/entropy_loss', entropy_loss, epoch)
            writer.add_scalar('train_gen/asr', asr, epoch)
            final_ims_gen = torchvision.utils.make_grid(final_ims_gen, normalize=True)
            writer.add_image('final_im_gen/{}'.format(epoch), final_ims_gen, epoch)
            

            scheduler.step()
            # time
            end_time = time.time()
            print(end_time - start_time)

            if epoch % args.test_freq == 0:
                torch.save(mp_generator.state_dict(), f'./model_zoo/DistGenerator_{args.task}_use_{args.model_name}_surrogate_lr{args.lr_gen}_lamba{args.lamba}_num_k{args.num_k}_sticker_{args.sticker_id}.ckpt')

            if epoch == epochs-1:
                print('begin black attack:')
                all_models = ['facenet', 'arcface', 'cosface', 'sphereface_50', 'mobilefacenet']
                for black_model_name in all_models:
                    print(f'attack balck model:{model_name}')
                    if black_model_name == 'facenet':
                        black_img_size = [160, 160]
                    elif black_model_name == 'mobilefacenet':
                        black_img_size = [112, 96]
                    else:
                        black_img_size = [112, 112]
                    trans_to_black_input = transforms.Resize(black_img_size)

                    with torch.no_grad():
                        print('##########DOP-RD##############')
                        average_query, average_success = Random_Search(args, model, mp_generator, dataloader, args.device, img_size, trans_to_black_input)
                        print('average_query:', average_query)
                        print('average_success:', average_success)
                        print('##########DOP-LO##############')
                        average_query, average_success = Bayesian_Optim(model, mp_generator, dataloader, args.device, img_size, trans_to_black_input)
                        print('average_query:', average_query)
                        print('average_success:', average_success)
                    print('##########DOP-DT##############')
                    average_query, average_success = NES_Optim(model, mp_generator, dataloader, args.device, img_size, trans_to_black_input)
                    print('average_query:', average_query)
                    print('average_success:', average_success)



def main():
    args = get_args()
    para = {'exp': args.exp, 'beta': args.beta, 'lr_gen': args.lr_gen,
            'epochs': args.epochs, 'alpha': args.alpha, 'patch_size': args.patch_size, 
            'dataset': args.dataset, 'k':args.num_k, 'lr_step':args.lr_step, 'lamba':args.lamba, 'task':args.task, 'model_name':args.model_name, 'MC_sample':args.MC_sample}
    writer, base_dir = get_log_writer(para)


    # seeting: load attack model、generator、dataset
    # img_size -----------------------------------------------------------------------------------------------------
    if para['model_name'] == 'facenet':
        input_size = [160, 160]
    elif para['model_name'] == 'mobilefacenet':
        input_size = [112, 96]
    else:
        input_size = [112, 112]

    # data -----------------------------------------------------------------------------------------------------
    if para['dataset'] == 'vggface':
        dataloader, dataloader_val, dataloader_all, id_num = load_facedata_unnormalized(args.batch_size, args.data_path, input_size, args.task)
        normalize_func = normalize_vggface
    elif para['dataset'] == 'imagenet':
        dataloader, dataloader_val = load_imagenet_unnormalize(args.batch_size, args.data_path)
        normalize_func = normalize_imagenet

    # attack model -----------------------------------------------------------------------------------------------------
    if para['task'] == 'cls':
        model_train = load_cls_model(model_name=args.model_name, device=args.device, id_num=id_num)
    elif para['task'] == 'reg':
        model_train = load_reg_model(model_name=args.model_name, device=args.device)
    
    print(f"use {args.model_name} as surrogate white-box model")

    # generator -----------------------------------------------------------------------------------------------------
    if args.mp_arc == 'res':
        mp_generator = load_generator(input_size, para['k'], 3, 64, init_type='kaiming' ,arc='res', n_blocks=args.mp_blocks).to(args.device)
        if args.only_black_optim:
            pretrained_dict = torch.load(args.G_path)
            mp_generator.load_state_dict(pretrained_dict)
    
    elif args.mp_arc == 'res-ir':
        mp_generator = load_generator(input_size, para['k'], 3, 64).to(args.device)
        pretrained_dict = torch.load(args.G_path)
        if args.only_black_optim:
            mp_generator.load_state_dict(pretrained_dict)
        else:
            mp_generator.load_state_dict(pretrained_dict, strict=False)

    # training setting -----------------------------------------------------------------------------------------------------
    optimizer_gen = torch.optim.Adam([
        {'params': mp_generator.parameters(), 'lr': para['lr_gen']}
    ], lr=0.1, betas=(0.5, 0.9))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer_gen, step_size=para['lr_step'], gamma=0.2)
    
    if para['task'] == 'cls':
        criterion = torch.nn.CrossEntropyLoss()
    elif para['task'] == 'reg':
        criterion = torch.nn.CosineSimilarity()
    
    # train and test
    DOPatch(dataloader, dataloader_val, dataloader_all, model_train, mp_generator, optimizer_gen, scheduler,
          criterion, para['epochs'], para['beta'], para['alpha'], para['task'], normalize_func, writer, args.device, input_size, id_num)


if __name__ == '__main__':
    main()