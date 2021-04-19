#!/bin/bash

os_name() {
    if [ ! -f $1 ];then
        HostOsName=unknown
        HostOsVersion=unknown
        return
    fi

    HostOsName=$(cat $1 | grep ^NAME= | awk -F "[\" ]" '{print $2}')
    if [ x"$HostOsName" = "x" ];then
        HostOsName=unknown
    fi
    HostOsVersion=$(cat $1 | grep ^VERSION_ID= | awk -F "\"" '{print $2}')
    if [ x"$HostOsVersion" = "x" ];then
        HostOsVersion=unknown
    fi
    return
}

get_os_info() {
    if [ -f /etc/os-release ];then
        os_name /etc/os-release
    elif [ -f /etc/centos-release ];then
        HostOsName=CentOS
        HostOsVersion=$(cat /etc/centos-release | awk '{print $4}')
    else
        which lsb_release >/dev/null 2>&1
        if [ $? -eq 0 ];then
            HostOsName=$(lsb_release -si)
            HostOsVersion=$(lsb_release -sr)
        else
            os_name /etc/issue
        fi
    fi
    return
}

get_system_info() {
    get_os_info
    HostArch=$(uname -m)
    KernelVersion=$(uname -r)
}

function version_gt() {
    if [ "$2"x = "x" ];then
        return 0
    else
        test "$(echo "$@" | tr " " "\n" | sort -V | head -n 1)" != "$1"
    fi
}
function version_le() {
    if [ "$2"x = "x" ];then
        return 0
    else
        test "$(echo "$@" | tr " " "\n" | sort -V | head -n 1)" == "$1"
    fi
}
function version_lt() {
    if [ "$2"x = "x" ];then
        return 0
    else
        test "$(echo "$@" | tr " " "\n" | sort -rV | head -n 1)" != "$1"
    fi
}
function version_ge() {
    if [ "$2"x = "x" ];then
        return 0
    else
        test "$(echo "$@" | tr " " "\n" | sort -rV | head -n 1)" == "$1"
    fi
}

version_vaild() {
    local ver_range=$1          #version range
    local ver="$2"              #version provider
    local clean_ver_range=$(echo "$ver_range" | sed 's/"//g')   # clean ver_range,去除引号啥的
    local new_ver_range=$(echo "$clean_ver_range" | sed 's/\[//' | sed 's/\]//' | sed 's/(//' | sed 's/)//')  #处理过的version range,去除[],() 得到类似 1.0,2.0

    local start=$(echo $new_ver_range | awk -F ',' '{print $1}')
    local end=$(echo $new_ver_range | awk -F ',' '{print $2}')

    if  echo $clean_ver_range | grep -Eq "^\[.*\]$" ; then
        # 闭合区间 [a,b]={x|a<=x<=b}
        if  version_ge $ver $start  && version_le $ver $end ; then
            # pass
            return 0
        fi
    elif echo $clean_ver_range | grep -Eq "^\[.*\)$"; then
        # 左闭右开 [a,b)={x|a<=x<b}
        if  version_ge $ver $start  && version_lt $ver $end ; then
            return 0
        fi
    elif echo $clean_ver_range | grep -Eq "^\(.*\]$"; then
        # 左开右闭 (a,b]={x|a<x<=b}
        if  version_gt $ver $start  && version_le $ver $end ; then
            return 0
        fi
    elif echo $clean_ver_range | grep -Eq "^\(.*\)$"; then
        # 开区间 (a,b)={x|a<x<b}
        if  version_gt $ver $start  && version_lt $ver $end ; then
            return 0
        fi
    else
        # 兼容老版本,ver_range=require_ver,ver >= require_ver 结果pass 否则 nopass。即依赖包的当前版本大于等于需求
        if version_ge $ver $clean_ver_range ;then
            return 0
        fi
    fi
    #not pass
    return 1
}

check_pkg_ver_deps() {
    ver_path=$1
    req_pkg=$2
    req_ver_path=$3
    local IFS=$' \t\n'

    ver_info_list=($(awk -F '[_=]' '$1=="required" && $2=="'$req_pkg'" {print $1";"$2";"$3";"$4";"$5}' "$ver_path"))

    for line in "${ver_info_list[@]}"
    do
        itf=`echo $line | cut -d";" -f 3`
        ver=`echo $line | cut -d";" -f 5`
        req_ver=`awk -F '=' '$1=="'$itf'_version" {print $2}' "$req_ver_path"`
        if ! version_vaild $ver $req_ver;then
            VerCheckStatus=FAIL
            return
        fi
    done
    VerCheckStatus=SUCC
}

check_install_path_valid() {
    local install_path="$1"
    # 黑名单设置，不允许//，...这样的路径
    if echo "${install_path}" | grep -Eq '\/{2,}|\.{3,}'; then
        return 1
    fi
    # 白名单设置，只允许常见字符
    if echo "${install_path}" | grep -Eq '^\~?[a-zA-Z0-9./_-]*$'; then
        return 0
    else
        return 1
    fi
}