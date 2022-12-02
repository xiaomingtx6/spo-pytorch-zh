# -*- coding: utf-8 -*-
import os
import copy
import logging
from utils import *
from models import *
from torch.utils.data import DataLoader, RandomSampler

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)


def save_model(opt, model, global_step):
    output_dir = os.path.join(opt.output_dir, 'checkpoint-{}'.format(global_step))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # take care of model distributed / parallel training
    model_to_save = (
        model.module if hasattr(model, "module") else model
    )
    logger.info(f'Saving model & optimizer & scheduler checkpoint to {output_dir}')
    torch.save(model.state_dict(), os.path.join(output_dir, 'model.pt'))


def train_base(opt):
    """加载数据"""
    hanspo = HANSPO()
    train_raw_examples = hanspo.read_jsons(opt.train_path)
    train_examples = hanspo._example_generator(train_raw_examples, "train", 512)
    train_datasets = Datasets(train_examples, "train", 1)

    dev_raw_examples = hanspo.read_jsons(opt.dev_path)
    dev_examples = hanspo._example_generator(dev_raw_examples, "train", 512)
    dev_datasets = Datasets(dev_examples, "train", 1)
    model = SOModel(opt.bert_dir, 0.1, True)
    model.to('cuda')
    training(opt, model, train_datasets, dev_datasets)


def training(opt, model, train_dataset, dev_datasets):
    swa_raw_model = copy.deepcopy(model)
    fgm = FGM(model=model)
    train_sampler = RandomSampler(train_dataset)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=opt.train_batch_size,
                              shuffle=True)

    dev_loader = DataLoader(dataset=dev_datasets,
                            batch_size=opt.train_batch_size,
                            shuffle=True)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    t_total = len(train_loader) * opt.train_epochs

    optimizer, scheduler = build_optimizer_and_scheduler(opt, model, t_total)
    logger.info("***** Running training *****")
    logger.info(f"  Num Examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {opt.train_epochs}")
    logger.info(f"  Total training batch size = {opt.train_batch_size}")
    logger.info(f"  Total optimization steps = {t_total}")

    global_step = 0

    model.zero_grad()

    save_steps = t_total // opt.train_epochs
    eval_steps = save_steps

    logger.info(f'Save model in {save_steps} steps; Eval model in {eval_steps} steps')

    log_loss_steps = 20

    avg_loss = 0.
    for epoch in range(opt.train_epochs):
        logger.info(f"  Num Epochs = {epoch}.............")

        for step, batch_data in enumerate(train_loader):
            model.train()

            for key in batch_data.keys():
                batch_data[key] = batch_data[key].to(device)
                # print(batch_data[key].device)
            #  普通loss传播
            loss = model(**batch_data)[0]
            loss.backward()
            # 使用fgm进行对抗训练
            fgm.attack()
            loss_adv = model(**batch_data)[0]
            loss_adv.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), opt.max_grad_norm)

            optimizer.step()
            scheduler.step()
            model.zero_grad()

            global_step += 1

            if global_step % log_loss_steps == 0:
                avg_loss /= log_loss_steps
                logger.info('Step: %d / %d ----> total loss: %.5f' % (global_step, t_total, avg_loss))
                avg_loss = 0.
            else:
                avg_loss += loss.item()

            if global_step % save_steps == 0:
                evalation(model, dev_loader, device)
                save_model(opt, model, global_step)

        # clear cuda cache to avoid OOM
    torch.cuda.empty_cache()
    logger.info('Train done')


def evalation(model, data_loader, device):
    model.eval()
    loss_total, precision_, recall_, f1_ = 0.0, 0.0, 0.0, 0.0
    total_step = 0
    for step, batch_data in enumerate(data_loader):
        total_step += 1
        for key in batch_data.keys():
            batch_data[key] = batch_data[key].to(device)

        lables = batch_data["labels"]
        loss, tmp_out = model(**batch_data)
        loss_, precision, recall, f1 = ComputeScore(tmp_out, lables.float(), 0.5, loss.item())
        loss_total += loss_
        precision_ += precision
        recall_ += recall
        f1_ += f1
    logger.info("Evaluation precision: %.5f, recall: %.5f, F1: %.5f, dev loss: %.5f"
                % (precision_ / total_step, recall_ / total_step, f1_ / total_step, loss_total / total_step))
