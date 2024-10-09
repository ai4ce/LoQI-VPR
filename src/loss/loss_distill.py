import torch
import torch.nn as nn
import torch.nn.functional as F

class Loss_distill(nn.Module):
    def __init__(self):
        super().__init__()

    def similarity(self,feature):
        b,c = feature.shape[:2]
        feature_flatten=feature.view(b,c,-1)
        feature_flatten=F.normalize(feature_flatten, p=2, dim=2)
        similarity_matrix=F.normalize(torch.bmm(feature_flatten, feature_flatten.transpose(1, 2)), p=2, dim=2)
        return similarity_matrix

    def forward(self,student_feature,teacher_feature):
        student_similarity=self.similarity(student_feature)
        teacher_similarity=self.similarity(teacher_feature)
        # b=teacher_similarity.size(0)
        # loss=self.MSE(student_similarity,teacher_similarity)*b
        loss=torch.norm(student_similarity-teacher_similarity,p='fro')
        return loss


