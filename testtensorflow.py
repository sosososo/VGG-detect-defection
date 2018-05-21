from tensorboardX import SummaryWriter

writer =SummaryWriter('log')
writer.add_scalar('T/L',1,1)
writer.add_scalar('T/L',2,2)
writer.add_scalar('T/L',3,3)
writer.add_scalar('T/L',4,4)
writer.add_scalar('T/L',5,5)
writer.add_scalar('T/L',65,6)
writer.add_scalar('T/L',7,7)

writer.export_scalars_to_json("./all_scalars.json")
writer.close()