class Solution:
    def maxSubArray(self, nums):
        i,j = 0,1
        max_ = -100
        while i<j and j<len(nums):
            max_ = max(max_,sum(nums[i:j]))
            if sum(nums[i+1:j])>sum(nums[i:j]) and i+1!=j:
                i+=1
            elif sum(nums[i:j+1])>sum(nums[i:j]):
                j+=1
        return max_

if __name__ == '__main__':
    so = Solution()
    res = so.maxSubArray([-2,1,-3,4,-1,2,1,-5,4])
    print(res)