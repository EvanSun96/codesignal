
import collections
import sys
import heapq

# 1. zigzag
def isZigZag(numbers):
    if not numbers or len(numbers) < 3:
        return []
    res = []
    for i in range(1, len(numbers)-1):
        if numbers[i-1]<numbers[i]>numbers[i+1] or numbers[i-1]>numbers[i]<numbers[i+1]:
            res.append(1)
        else:
            res.append(0)
    return res

# 2. pattern matching
def binaryPatternMatching(pattern, s):
    # if k small, can just take O(nk) to loop through and check 
    # o.w. use bfs
    if not pattern or not s or len(pattern) > len(s):
        return 0

    pattern = [int(x) for x in pattern]
    vowels = set({'a','e','i','o','u','y'})
    bin_s = [1]*len(s)
    q = []
    first_0 = False
    if pattern[0] == 0:
        first_0 = True
    for i, ch in enumerate(s):
        if ch in vowels:
            bin_s[i] = 0
            if first_0:
                q.append(i)
        elif not first_0:
            q.append(i)
    cnt = 0
    level = 1
    while q and level < len(pattern):
        sz = len(q)
        for j in range(sz):
            candidate = q.pop(0)
            if candidate + 1 < len(bin_s) and pattern[level] == bin_s[candidate+1]:
                q.append(candidate+1)
                if level + 1 == len(pattern):
                    cnt += 1
        level += 1
    return cnt 

    # run time fast enough ? better solution ? correctness - test more cases ? 

# 3. sort matrix
def sortMatrix_two_ascending_criteria(m):
        if not m:
            return []

        cnts = collections.defaultdict(int)
        for i in range(len(m)):
            for j in range(len(m[0])):
                cnts[m[i][j]] += 1
        
        colle_cnts = collections.defaultdict(list)
        for k in cnts.keys():
            colle_cnts[cnts[k]].append(k)
        
        sort_m = []
        for k in sorted(colle_cnts.keys()):
            for v in sorted(colle_cnts[k]):
                sort_m.append(v)

        q = [(len(m)-1, len(m)-1)]
        idx = 0
        while q and idx < len(sort_m):
            x, y = q.pop(0)
            m[x][y] = sort_m[idx]
            cnts[sort_m[idx]] -= 1
            if cnts[sort_m[idx]] == 0:
                idx += 1
            if x == 0 and y == 0:
                break 
            if y-1>=0 and x +1 == len(m):
                # append left child
                q.append((x, y-1))
            if x-1>=0:
                q.append((x-1, y))
            
        return m 

        # better sorting procedure ? MLE won't be a problem ? 

# 4. check ABC
# check if all b's elements in c
# possible TLE: 
# bfs or dfs to find all longest subarray of a that contain only elements in c
# check if b is in these results 
# improve:
# bfs or dfs to find length of the longest subarray containing only c elements
# check if b length right and b subarray in c (with bfs O(n^2) ) 
def longestSubarrayCheck(a, b, c): 

    # consider empty array cases ? 
    
    c = set(c)
    for bi in b:
        if bi not in c:
            return False
    mx = 0 
    i = 0
    for j in range(len(a)):
        if a[j] not in c:
            mx = max(mx, j-i)
            i = j + 1
        elif j == len(a) -1:
            mx = max(mx, len(a)-i)
    if mx != len(b):
        return False

    # check if b is a continuous subarray of a 
    start = b[0]
    q = []
    for i, ai in enumerate(a):
        if ai == start:
            q.append(i)

    level = 1
    while q and level < len(b):
        sz = len(q)
        for i in range(sz):
            candidate = q.pop(0)
            if candidate + 1 < len(a) and a[candidate+1] == b[level]:
                if level + 1 == len(b):
                    return True 
                q.append(candidate+1)   
        level += 1
    return False 

    # correctness and runtime ok ? ? 

# 5. Reverse Number
def reverseDigitsInPairs(n):

    digits = []
    value = n
    while value: 
        rem = value % 10 
        value = value // 10 
        digits.append(rem)
    
    digits.reverse()
    i = 0

    end = len(digits) - 1
    if len(digits) % 2 == 1:
        end -= 1

    while i + 1 <= end:
        temp = digits[i]
        digits[i] = digits[i+1]
        digits[i+1] = temp 
        i += 2

    res = 0 
    multiplier = 1
    for i in range(len(digits)-1, -1, -1):
        res += digits[i] * multiplier 
        multiplier *= 10 
    return res 

# 6. occurence sequence
def maxKOccurences(sequence, words):
    if not sequence:
        return []
    res = []
    for word in words: 
        if not word:
            res.append(0)
            continue
        idx = 0
        cnt = 0 
        maxcnt = 0 
        for j in range(len(sequence)):
            if word[idx] != sequence[j]:
                maxcnt = max(cnt, maxcnt)
                cnt = 0 
                idx = 0 
            else:
                idx += 1
                if idx == len(word):
                    idx = 0
                    cnt += 1
                    if j == len(sequence) - 1:
                        maxcnt = max(cnt, maxcnt)
        res.append(maxcnt)

    return res

    # try more test cases ? 



# 7 sortMatrix_chestboard
def sortChessSubsquares(numbers, queries):
    if not numbers or not queries:
        return numbers
    black = []
    white = []
    m, n = len(numbers), len(numbers[0])
    for query in queries: 
        # constraint check, if subsquare matrix not enough in numbers, sort partially ?
        x, y, w = query[0], query[1], query[2]
        if 0<=x<m and 0<=y<n:
            for i in range(w):
                for j in range(w):
                    nx, ny = x+i, y+j
                    if 0<=nx<m and 0<=ny<n:
                        if nx % 2 == 0 and ny % 2 == 0 or nx %2 == 1 and ny % 2 == 1:
                            black.append(numbers[nx][ny])
                        else:
                            white.append(numbers[nx][ny]) 
            black.sort()
            white.sort()
            for i in range(w):
                for j in range(w):
                    nx, ny = x+i, y+j
                    if 0<=nx<m and 0<=ny<n:
                        if nx % 2 == 0 and ny % 2 == 0 or nx %2 == 1 and ny % 2 == 1:
                            numbers[nx][ny] = black.pop(0)
                        else:
                            numbers[nx][ny] = white.pop(0)
    return numbers

    # runtime max(w)^2 * k queries - potentially too slow ? 
    # correctness, try more cases ?

# 8. matrix query ... deactivate cells
def matrixQueries(n, m, queries):
    r = [1]*(n+1)
    c = [1]*(m+1)
    res = []
    for query in queries:
       if len(query) == 2:
            if query[0] == 1:
               # d r
                r[query[1]] = 0
            else:
                c[query[1]] = 0
    minr, minc = n, m
    for i in range(1, n+1):
        if r[i] == 1:
            minr = i
            break
    for i in range(1, m+1):
        if c[i] == 1:
            minc = i 
            break
    for i in range(len(queries)-1, -1, -1):
        query = queries[i]
        if query == [0]:
            res.append(minr*minc)
        elif query[0] == 1:
            ri = query[1]
            r[ri] =  1
            if ri < minr:
                minr = ri
        else:
            ci = query[1]
            c[ci] = 1
            if ci < minc:
                minc = ci
    res.reverse()
    return res
    # correctness ? 


         


 

# 9 numberSigningSum
def numberSigningSum(n):

    # see whether additional constraints

    res = 0
    value = n 
    pos = True 
    if len(str(n)) % 2 == 0:
        pos = False
    while value:
        digit = value % 10 
        value = value // 10 
        if pos:
            res += digit 
        else:
            res -= digit
        pos = not pos 
    return res 

# 10 sawtooth
def countSawSubarrays(array):
    if not array or len(array) < 2:
        return 0
    n = len(array)
    large_th_pre = [0]*n
    small_th_pre = [0]*n

    res = 0
    for i in range(1, len(array)):
        num = array[i]
        if num == array[i-1]:
            continue
        elif num > array[i-1]:
            subcnt = 1
            if i -2 >= 0 and small_th_pre[i-1] > 0:
                subcnt += 1
                subcnt += large_th_pre[i-2]
            large_th_pre[i] = subcnt
            res += large_th_pre[i]
        else: 
            subcnt = 1
            if i-2 >= 0 and large_th_pre[i-1] > 0:
                subcnt += 1
                subcnt += small_th_pre[i-2]
            small_th_pre[i] = subcnt
            res += small_th_pre[i]
    return res

    # correctness ? test more cases

# 11. check 2, 4, 0
def countOccurrences(n):
    cnts = [0]*(n+1)
    cnts[2] = cnts[4] = cnts[0] = 1
    for v in range(10, len(cnts)):
        r = v % 10 
        d = v // 10 
        curcnt = cnts[d] 
        if r in [0, 2, 4]:
            curcnt += 1
        cnts[v] = curcnt 
    print(cnts)
    return sum(cnts)




# 12 concat swaps
def concatSwaps(input_s, size):
    # check constraints if size doesn't sum to len(input_s)  !
    if not input_s or not size or len(size) == 1:
        return input_s
    concats = []
    start = 0
    for s in size:
        substr = input_s[start: start+s]
        concats.append(substr)
        start = start + s

    for i in range(1, len(concats), 2):
        temp = concats[i-1]
        concats[i-1] = concats[i]
        concats[i] = temp 
        if i + 2 == len(concats):
            break
    return ''.join(concats)

# 13 bouncing matrix 
# calculate two directions separately then sum then hashmap sorted two things accordingly


# 14 power two  
def powerTwo(a):
    mymap = collections.defaultdict(int)
    powers2 = collections.defaultdict(int)
    for x in a:
        mymap[x] += 1
    
    ans = 0
    for x in a:
        y = nextpower2(x) - x
        if x == 0:
            continue
        if y == 0:
            powers2[x] += 1
        ans += mymap[y]
    
    for val in powers2.values():
        ans += val*(val+1) / 2
    return ans
# ?  https://leetcode.com/discuss/interview-question/808272/Robinhood-OA

def nextpower2(v):
    v -= 1
    v |= v >> 1
    v |= v >> 2
    v |= v >> 4
    v |= v >> 8
    v |= v >> 16
    return v + 1

# this comes DIRECTLY from LC solution ! 


# 15 peak 
def min_peaks(arr):
  n = len(arr)

  # add sentinel
  arr.append(float('-inf'))
  ans = []

  # represents doubly-linked list
  # 0 <-> 1 <-> ... <-> n <-> 0
  prv = [n] + range(n)
  nxt = range(1, n+1) + [0]

  # initialize min-heap of peaks
  peaks = []
  def _check_peak(i):
    if arr[prv[i]] < arr[i] > arr[nxt[i]]:
      heapq.heappush(peaks, (arr[i], i))
  for i in xrange(n):
    _check_peak(i)

  # remove min peak and check neighbors
  while peaks:
    val, i = heapq.heappop(peaks)
    ans.append(val)
    # before: j <-> i <-> k
    j, k = prv[i], nxt[i]
    prv[k] = j
    nxt[j] = k
    # after: j <-> k
    _check_peak(j)
    _check_peak(k)

  # remove sentinel (optional)
  arr.pop()
  return ans

  # this comes DIRECTLY from LC solution ! 


# 16 stock 

def maxRevenueFromStocks(prices, algo, k):
    if not prices:
        return 0
    # mk sure k <= len(prices) ** 
    runSum = 0
    n = len(algo)
    for i in range(n):
        if algo[i] == 0:
            runSum -= prices[i]
        elif algo[i] == 1:
            runSum += prices[i]
    for j in range(k):
        if algo[j] == 0:
            runSum += 2 * prices[j]

    mx = runSum 
    for j in range(k, n):
        print(runSum)
        if algo[j] == 0:
            runSum += 2 * prices[j]
        if algo[j-k] == 0:
            runSum -= 2 * prices[j-k]
        mx = max(mx, runSum)
    return mx 
    # check correctness ? 

# 17
def newTextEditor(operations):
    strBuilder = ''
    lastCopied = ''
    lastOp = []
    lastString = []

    for op in operations:
        # print(operations)
        parts = op.split(" ")
        command = parts[0]
        if command == 'DELETE':
            if strBuilder:
                lastOp.append(command)
                lastString.append(strBuilder[len(strBuilder)-1])
                strBuilder = strBuilder[: len(strBuilder)-1]

        elif command == 'PASTE':
            if lastCopied != '':
                strBuilder += lastCopied
                lastOp.append(command)
                lastString.append(lastCopied)
        elif command == 'UNDO':
            # lastOp, lastString, strBuilder
            if lastOp[-1] == 'DELETE':
                strBuilder += lastString[-1]
            elif lastOp[-1] == 'INSERT' or lastOp[-1] == 'PASTE':
                strBuilder = strBuilder[:len(strBuilder)-len(lastString[-1])]
            lastOp.pop(-1)
            lastString.pop(-1)
        elif command == 'INSERT':
            if len(parts) == 2:
                strBuilder += parts[1]
                lastOp.append(parts[0])
                lastString.append(parts[1])
        elif command == 'COPY':
            if len(parts) == 2:
                idx = int(parts[1])
                lastCopied = strBuilder[idx:]

    return strBuilder

# 18 coolfeature of new programming
# The problem looks fairly straightforward and is basically an extension of 2 sum problem. 
# I suggest you to use 2 Hashmaps. Firstly, scan through all elements of a and put them inside a Hashmap. 
# Also, keep all the keys of first map into a new array 'arr'. Then use a second map for the b array as well. 
# Then, iterate over all the queries and they are basically of 2 kinds, one has size 2 and the other 3. 
# Whenever the query has size 2,scan through all the elements of 'arr' and check if x-arr[i] is present in second hashmap. 
# This lookup can easily be done in constant time. If such an element is found then simply, 
# get the value of arr[i] from first map and the value of x-arr[i] from the second map and multiply them. 
# Add this to your total count. Repeat this for all the queries of size 2. Now, when queries are of size 3, 
# simply get the b[i] element from b array. Look this up in the second map and find its value. Decrease this by 1 and update it in map. 
# Accordingly, put the new b[i] value which is x inside the second map, if key exists add 1 and update, if it does not simply put key as x and value as 1 in the map. 
# Also, overwrite the b[i] in the b array with x.

    
# 19 rotate matrix k times
# ommitted, below is k = 1 time, notice tho diagonals cannot be changed different !!!!
def rotate(matrix):
        """
        Do not return anything, modify matrix in-place instead.
        """
        if not matrix:
            return 
        for i in range(len(matrix)):
            for j in range(i + 1, len(matrix[0])):
                temp = matrix[i][j]
                matrix[i][j] = matrix[j][i]
                matrix[j][i] = temp 

        for i in range(len(matrix)):
            matrix[i].reverse()
        

# 20 max arithmatic length ?


# 21 broken key ommitted

# 22 pythagorean triple ommitted 

# 23 cannot use dp !! use bfs or keep upd nextv for new indexes .. check if can reach end (kind of brute force)
def shuffleThePieces(arr, pieces):
    mp = collections.defaultdict(list)
    for p in pieces:
        k = p[-1]
        mp[k].append(p)
    dp = [True]+[False]*len(arr) # dp[i+1] -> arr[i]
    for i in range(1, len(dp)):
        idx = i - 1
        if arr[idx] not in mp:
            continue
        cand = mp[arr[idx]]
        for p in cand:
            if i - len(p) >= 0 and dp[i-len(p)] and p == arr[idx-len(p)+1:idx+1]:
                dp[i] = True 
    print(dp)
    return dp[-1]



# 26 booleanDeque - [n, operations -> L, Cind ... ]
# track let most zero

# 29 reverse numbers then sum
def reverseSum(numbers):

    res = 0
    for number in numbers:
        number = [d for d in str(number)]
        i = 0
        j = len(number) -1
        while number[j] == '0':
            j -= 1
        while i <= j:
            temp = number[i]
            number[i] = number[j]
            number[j] = temp 
            i += 1
            j -= 1
        res += int(''.join(number))
    return res 

# 30 
def countStrElementCountOddZeros(inputs):
    cnt = 0
    for num in inputs:
        subcnt = num.count('0')
        if subcnt % 2 == 1:
            cnt += 1
    return cnt

# 31 array shift, two pointers fast enough ? or try bfs 

# 32 keep adding string's numbers by k group until sum length < k 
def addKGroupSumUntilSumLengthSmallerThanK(inputstr, k):

    curstr = inputstr
    while len(curstr) > k:
        group = len(curstr) // k 
        if len(curstr) % k != 0:
            group += 1
        itrvals = ['']*group
        for g in range(group):
            subsum = 0
            for i in range(k):
                if i +g*k == len(curstr):
                    break
                subsum += int(curstr[i+g*k])
            itrvals[g] = str(subsum)
        curstr = ''.join(itrvals)
    return curstr

    # correctness ? check more test cases and corner

# 35 is submatrix full, ignore maybe brute force

# 36 rotate and fall similar to lc move zeroes see thoughts   
# 类似 LeetCode 的 MoveZeroes，先找到所有的障碍物，然后 将 start，障碍物，end 之间所有的 interval 进行 move zeroes 的操作
# 把 box 移到最后， 再 rotate 就好了。
# move zeroes are the following 
# def moveZeroes(self, nums: List[int]) -> None:
#         """
#         Do not return anything, modify nums in-place instead.
#         """
#         n = len(nums)
#         nonzero = zero = 0

#         while nonzero < n and zero < n:
#             while nonzero < n and nums[nonzero] == 0:
#                 nonzero += 1
#             if nonzero == n:
#                 break
#             while zero < n and nums[zero] != 0:
#                 zero += 1
#             if zero < n:
#                 if nonzero > zero:
#                     nums[zero] = nums[nonzero]
#                     nums[nonzero] = 0
#                 else:
#                     nonzero = zero + 1 
        
# 39 3*n matrix sliding window
# 给一个3*n的matrix，用一个3*3的滑动窗口从左到右，判断每次圈住的9个数是否正 好是1-9。我用slice操作+set转换然后判断len==9来写，
# 很简便(i.e. len(set(m[0][i:i+3] + m[1][i:i+3] + m[2][i:i+3])) == 9)，very easy，3min速通

# 40 rev ai aj rev ... 
# => arr[x] - rev(arr[x]) equal then a pair 
def revcnt(a):
    cnts = collections.defaultdict(int)
    for num in a:
        rev = int(str(num)[::-1])
        cnts[num - rev] += 1
    res = 0
    for val in cnts.values():
        cur = val - 1
        # need remove duplicates or not ?
        while cur:
            res += cur
            cur -= 1
    return res

    # correctness ? 

# 41 rectangular figure fall
def figureEnderGravity(matrix):
    if not matrix or not matrix[0]:
        return matrix
    m, n = len(matrix), len(matrix[0])
    minstep = sys.maxsize
    for j in range(n):
        i = 0
        while i < m:
            while i < m and matrix[i][j] != 'F':
                i += 1
            if i == m:
                continue
            while i < m and matrix[i][j] == 'F':
                i += 1
            if i < m:
                cnt = 0
                while i < m and matrix[i][j] != '#':
                    i += 1
                    cnt += 1
                minstep = min(minstep, cnt)
    if minstep == sys.maxsize or minstep == 0:
        return matrix

    for i in range(m-1, minstep-1, -1):
        for j in range(n):
            if matrix[i-minstep][j] == 'F':
                matrix[i][j] = 'F'
                matrix[i-minstep][j] = '.'

    return matrix

# 43
# 第四题:地里原题，输入一个由一个一维数组形成的二维数组，要求把二维数组还原成一 维数组， 看例子: 原本有个一维数组[1, 2, 3, 4, 5, 6]，然后题目把他写成这个样子，
# 类 似邻接表[[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]]， 这个二维数组的每一行的0和1位置都是可以 交换的，二维数组的每一行也都是可以交换的，
# 也就是说给你的输入可能是[[6, 5], [2, 3], [4, 3], [4, 5], [1, 2]]. 要求还原为一维数组，可以是正序或者倒序# 
# 第四题提供一下简单思路。

# 44 newpaper alignment
# can't see all descriptions in full, assume align to left if words exceeded one whole line to the next line
def justifyNewspaperText(lines, aligns, width):

    res = [''.join(['*']*(width+2))]
    lines.reverse()
    stack = lines
    print(stack)
    i = 0
    multiline = 0
    while stack:
        line = stack.pop(-1)
        cur = ' '.join(line)
        l = len(cur)
        print(i)
        print(aligns[i])
        if l == width:
            res.append('*'+cur+'*')
            if multiline > 0:
                multiline -= 1

        elif l < width :
            empty = ''.join([' ']*(width-l))
            if aligns[i] == 'LEFT' :
                res.append('*'+cur+empty+'*')
            else:
                res.append('*'+empty+cur+'*')
            if multiline > 0:
                multiline -= 1

        else:
            chunks = []
            curnewline = [line[0]]
            cursum = len(line[0])
            if cursum == width:
                chunks.append(curnewline)
                cursum = 0
            for w in range(1, len(line)):
                if 1 + len(line[w]) + cursum > width:
                    chunks.append(curnewline)
                    curnewline = [line[w]]
                    cursum = len(line[w])
                else:
                    cursum += 1 + len(line[w])
                    curnewline.append(line[w])
            chunks.append(curnewline)
            multiline = len(chunks)
            for cidx in range(len(chunks)-1, -1, -1):
                stack.append(chunks[cidx])
            print(chunks)
        if multiline == 0:
            i += 1

        
    res.append(''.join(['*']*(width+2)))
    return res
                


# 45 minimize diff (abs) of ai and bi elements by replacing an ai
# 我刚开始没想清楚，就找到差得最远的a_i和b_i，然后将这个a_i替换为跟b_i差的最少的那个a_k，然而这个思路并不对。。。把a跟b都画在数轴上发现了解题思路，
# 就是对于每个b_i来说都去找离它最近的那个a_k，然后gain_i就是|b_i - a_i| - |b_i - a_k|，意为如果替换当前的b对应的a能达到的最大收益。
# 找到最大的gain，sum(diff)减去其即可。我用了二分去找离b_i最近的a_k，不清楚如果直接找会不会超时，不妨试试。


# 46 digit anagrams
def digitAnagrams(a):
    if not a or len(a) < 2:
        return 0
    mp = collections.defaultdict(int)
    for num in a:
        k = int(''.join(sorted(str(num))))
        mp[k] += 1
    res = 0
    for val in mp.values():
        cur = val - 1
        while cur:
            res += cur 
            cur -= 1
    return res

    # correctness ? 

# 47
# lbayrinthEscape
def hasValidPath(grid):
        m = len(grid)
        n = len(grid[0])
        
        queue = collections.deque([(0, 0)])
        
        directions = {1: [(0, 1), (0, -1)],
                      2: [(-1, 0), (1, 0)],
                      3: [(1, 0), (0, -1)],
                      4: [(1, 0), (0, 1)],
                      5: [(-1, 0), (0, -1)],
                      6: [(0, 1), (-1, 0)]}
        
        visited = [[0]*n for _ in range(m)]
        visited[0][0] = 1
        
        while queue:
            
            x, y = queue.popleft()
            if x == m-1 and y == n-1:
                return True
            
            dirc = directions[grid[x][y]]
            for i, j in dirc:
                xx, yy = x + i, y + j
                if 0 <= xx < m and 0 <= yy < n and visited[xx][yy] == 0: 
                    for ii, jj in directions[grid[xx][yy]]: # inverse is linked
                        if ii + i == 0 and jj + j == 0:
                            queue.append((xx, yy))
                            visited[xx][yy] = 1

        return False
    # DIRECTLY from LC

# 51
# def duplicatesOnSegment(arr) ? 
# 像 lc 395？


# 54 remove one digit how many smaller than such removal a smaller than b
# brute force O(n^2)

# 55 three operation on matrix 
def rotateMatrix(a):
  N = len(a)
  matrix = [ list(reversed([row[i] for row in a])) for i in range(N)]
  return matrix

def diagonalMain(a):
  matrix = [[row[i] for row in a] for i in range(len(a))]
  return matrix

def diagonalSecond(a):
  return list(reversed(rotateMatrix(a)))

def mutateMatrix(matrix, queries):
  for q in queries:
    if q == 0:
      matrix = rotateMatrix(matrix)
    elif q == 2:
      matrix = diagonalMain(matrix)
    elif q == 1:
      matrix = diagonalSecond(matrix)
  return matrix

# DIRECTLY from LC


# 57 className & methodName
def constructorNames(className, methodName):
    classCnts = [0]*26
    methodCnts = [0]*26
    if len(className) != len(methodName):
        return False
    for i in range(len(className)):
        classCnts[ord(className[i])-ord('a')] += 1
        methodCnts[ord(methodName[i])-ord('a')] += 1
    for i in range(26):
        if classCnts[i] > 0 and methodCnts[i] == 0 or classCnts[i] == 0 and methodCnts[i] > 0:
            return False
    classCnts.sort()
    methodCnts.sort()
    for i in range(26):
        if classCnts[i] != methodCnts[i]:
            return False
    return True

# 59 
# find longest prefix that is palindrome and cut from s, repeat until len(s) < 2
# palindromeCutting(s)
# SOL: use palindromic substring to find all palindromes in n^2, so each while loop takes O(1)
# following find palindromic substring sol, modify it
# def countSubstrings(self, s: str) -> int:
#         n = len(s)
#         dp = [[False] * n for _ in range(n)] 
#         cnt = 0
#         for i in range(n-1,-1,-1):
#             for j in range(i, n):
#                 if s[i]==s[j] and ((j-i+1)<3 or dp[i+1][j-1]):
#                     cnt += 1
#                     dp[i][j] = True
#         return cnt

# 61 concat edge letters, ommitted

# 62 three non-empty contiguous subarray: two methods:::: 

# def countSubsegments(arr):
#     n = len(arr)
#     if n <= 2:
#         return 0
#     total = sum(arr)
#     si = 0
#     count = 0
#     for i in range(n-2):
#         si += arr[i]
#         sj = 0
#         for j in range(i+1, n-1):
#             sj += arr[j]
#             if si <= sj <= total - si - sj:
#                 count += 1
#     return count 

    # DIRECTLY from lc, correctness not sure test more ? 
def countSubsegments(A) :
        count = 0
        S = sum(A)
        sum1 = 0
        for i in range(0, len(A) - 2):
            sum1 += A[i]
            sum2 = 0
            sum3 = S - sum1
            for j in range(i + 1, len(A) - 1):
                sum2 += A[j]
                sum3 -= A[j]
                if sum1 <= sum2 and sum2 <= sum3:
                    count += 1

        return count 
    # DIRECTLY from lc, correctness not sure test more ? 

# 63 lower upper find cnts of i, j
# sort + binary search
# check lc again:
# bLow is the lowest index in the bSquare array for which "low<= a_i + bLow <= high"
# Similarly bhigh is the highest index
# for each index a_i you have to find the leftMostIndex bLow such that b[bLow]^2 >= "lower -a_i^2", 
# and then you have to find the rightmost index bHigh such that b[bHigh]^2 <= "higher - a_i^2", 
# once you have this range [bLow, bHigh] for every a_i, you simply add (bHigh-bLow+1) to your ans. 
# computing bLow,bHigh can be done via binary search on the bSquare array

# 66
# 给你一堆由0和1组成的array，和query input，返回modified过的output matrix。
# input：
# array: [0,0,1,0,1,1,1] 
# query input: (0,2)
# query input: (1,2)
# query input:(0,4)
# 意思：找到array里连续两个0的segment，然后把他们变成1。跑完第一个query，array变[1,1,1,0,1,1,1]。
# 意思：找到array里连续两个1 的segment,然后把他们变成0.第三个query 忽略因为找不到……
# SOL: use two sets zeroes and ones to store first idx of 0's and 1's and how long they are 


# picture (4)
def removingPairsGame(numbers):
    # if not like adjacent numbers moved after some been popped, then don't even need others 
    # just count how many pairs of adjacent duplicates ! ! !
    stack = numbers
    cnt = 0
    while len(stack) > 1:
        helperstack = []
        while stack:
            cur = stack.pop(-1)
            if not helperstack or helperstack[-1] != cur:
                helperstack.append(cur)
            else:
                helperstack.pop(-1)
                cnt += 1
                if not stack:
                    break
                while helperstack:
                    stack.append(helperstack.pop(-1))
                break

    if cnt % 2 == 1:
        return 'Alice'
    return 'Bob'

# picture (5)
# 4 steps find first nonzero substract sum partially until all zeroes return sum in step 3
# sol: two while loops, one iterate until reach end of arr, 
# another caculates deduction of x for each element until the next leading zero 

if __name__ == '__main__':
    # numbers = [1, 2, 3]
    # print(isZigZag(numbers))

    # pattern = '011'
    # s = 'abecbihhcy'
    # pattern = '010'
    # s = 'amazing'
    # pattern = '11111'
    # s = 'mmmmmmmmmm'
    # print(binaryPatternMatching(pattern, s))

    # m = [ [1,2,1],[2,1,3],[3,4,5]]
    # m = [[1, 4, -2], [-2, 3, 4], [3,1, 3]]
    # print(sortMatrix_two_ascending_criteria(m))

    # a = [1, 1, 5, 1, 2]
    # b = [1, 2]
    # c = [2, 1]
    # a = [1, 3, 7, 5, 6, 3, 6, 1,3, 4,3, 5, -1, 1, 2]
    # b = [1, 3, 4, 3, 5]
    # c = [1, 3, 4, 5]
    # print(longestSubarrayCheck(a, b, c))

    # n = 123456
    # n = 72328
    # n = 2
    # print(reverseDigitsInPairs(n))

    # sequence = "ababcbabc"
    # words = ["ab", "babc", "bca"]
    # sequence = "dabcacab"
    # words = ["dabcacab"]
    # print(maxKOccurences(sequence, words))
    
    # n = 52134
    # n = 12345
    # n = 104956
    # print(numberSigningSum(n))

    # numbers = [[1, 4, 3, 2], [8, 4, 7, 1], [1, 5, 2, 1]]
    # queries = [[0, 1, 3], [1, 0, 2]]
    # numbers = [[5, 4], [1, 3], [2, 3]]
    # queries = [[0,0,2],[1,0,2],[0,0,2]]
    # print(sortChessSubsquares(numbers, queries))

    # array = [9, 8, 7,  6, 5]
    # array = [10,10,10]
    # array = [3, 2, 4, 5, 6, 4, 7, 7, 3]
    # print(countSawSubarrays(array))

    # input_s = "descognail"
    # size = [3, 2, 3, 1, 1]
    # size = [3, 2, 3, 2]
    # size = [10]
    # print(concatSwaps(input_s, size))

    # numbers = [123, 456, 1100]
    # numbers = [18500]
    # numbers = [1200]
    # print(reverseSum(numbers))

    # inputs = ["1","2070", "30", "0", "111"]
    # print(countStrElementCountOddZeros(inputs))

    # inputstr = '1111122222'
    # k = 3
    # inputstr = '11'
    # print(addKGroupSumUntilSumLengthSmallerThanK(inputstr, k))

    # a = [25, 35, 872, 228, 53, 278, 872]
    # a = [25, 52, 11, 11, 13]
    # print(digitAnagrams(a))

    # a = [11, 11]
    # print(revcnt(a))

    # arr = [1, 2,2, 2, 5, 0]
    # arr = [1, 1, 1]
    # arr = [1, 2, 0]
    # print(countSubsegments(arr))

    # matrix = [['F', 'F', 'F'], ['.','F','.'], ['.','F','F'],['#','F','.'],['F','F','.'],['.','.','.'],['.','.','#'],['.','.','.']]
    # matrix = [['F', 'F', 'F'], ['.','F','.'], ['.','F','F'],['#','F','.'],['F','F','.'],['.','.','.']]
    # matrix = [['F', 'F', 'F'], ['#','F','.'], ['.','F','F'],['#','F','.'],['F','F','.'],['.','.','.']]
    # matrix = [['F', 'F', 'F'], ['.','F','.'], ['.','F','F'],['#','F','.'],['F','F','.'],['.','.','.'],['#','.','#'],['.','.','.']]
    # matrix = [['.', 'F', 'F']]
    # print(figureEnderGravity(matrix))

    # lines = [['hello','world'],['How','areYou','doing'],['Please look', 'and align','to right']]
    # aligns = ['LEFT', 'RIGHT', 'RIGHT']
    # lines = [['hello','world'],['How','areYou','doing'],['Please look', 'and align','to right'], ["thanks"]]
    # aligns = ['LEFT', 'RIGHT', 'RIGHT', 'LEFT']
    # width = 16
    # lines = [['hello'],['hi', 't'],['hi', 't', 'I', 'am', 'ok']]
    # aligns = ['LEFT', 'RIGHT','LEFT']
    # wdith = 5
    # print(justifyNewspaperText(lines, aligns, wdith))

    # operations = ['INSERT Code', 'INSERT Signal', 'DELETE', 'UNDO']
    # operations = ['INSERT Da', 'INSERT Da','COPY 0', 'UNDO', 'UNDO', 'INSERT Da','COPY 0', 'UNDO', 'PASTE', 'PASTE', 'COPY 2', 'PASTE', 'PASTE', 'DELETE', 'INSERT aaam']
    # print(newTextEditor(operations))

    # numbers = [1, 5, 1, 4, 4, 5, 5, 6, 6, 1, 5]
    # numbers = [1, 5, 1, 1, 1]
    # print(removingPairsGame(numbers))

    # className = 'abbzccc'
    # methodName = 'babzzcz'
    # print(constructorNames(className, methodName))

    # n = 3
    # m = 4
    # queries = [[0],[1,2],[0],[2,1],[0],[1,1],[0]]
    # print(matrixQueries(n, m, queries))

    # arr = [1, 2, 5, 3]
    # pieces = [[5], [1, 2], [3]]
    # arr = [1, 2, 5, 3, 6]
    # pieces = [[1, 2], [5], [6, 3]]
    # arr = [1, 2, 5, 5, 5, 3, 6, 1, 5,5, 5]
    # pieces = [[1, 2], [5, 5, 5], [3, 6, 1]]
    # print(shuffleThePieces(arr, pieces))

    # n = 10
    # print(countOccurrences(n))

    # prices = [2, 4, 1, 5, 2, 6, 7]
    # algo = [0, 1, 0, 0, 1, 0, 0]
    # k = 4
    # print(maxRevenueFromStocks(prices, algo, k))

    a = [0,2,5]
    a = [2, 3, 5, 4]
    a = [3, 2, 1, 0, 1]
    a = [3, 4, 4, 4, 0, 6, 0, 2, 0, 4]
    a = [18, 73, 1, 14, 34, 1, 75, 43, 68, 90, 44]
    print(powerTwo(a))