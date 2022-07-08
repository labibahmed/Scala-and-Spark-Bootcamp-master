def isEven(num:Int):Boolean= {return num%2==0}

println(isEven(4))

def checkList(list:List[Int]):Boolean ={
    for(num <- list){ 
        if(isEven(num)) return true}
    return false
}
val list = List(5,5,6,15)
println(checkList(list))

def luckySeven(list:List[Int]):Int={
    var x = list.sum
    for (num <- list){
        if(num == 7) x += 7
        }
    return x
}

println(luckySeven{list})

def isBalance(list:List[Int]):Boolean={
    var x = 0
    var y = 0
    for(num <- list){
        x += num
        y = list.sum - x
        if (x == y) return true
    }
    return false
}

println(isBalance(list))

def isPalindrome(word:String):Boolean={return word == word.reverse}

println(isPalindrome("poop"))