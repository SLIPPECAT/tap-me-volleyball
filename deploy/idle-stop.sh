#!/bin/sh
# 쓸 때만 켜는 서버(10/31): 30분 넘게 요청이 없고 도는 작업이 없으면 VM을 끈다. 5분마다 systemd 타이머가 부른다.
# 꺼 두면 계산 요금이 나가지 않고, 사용자가 안내 페이지(deploy/wake)에 오면 다시 켜진다.
# 심사 기간처럼 계속 켜 둘 때는: sudo touch /etc/tapme-keep-on (지우면 다시 자동으로 꺼짐)
[ -e /etc/tapme-keep-on ] && exit 0
r=$(curl -s --max-time 5 http://127.0.0.1:8000/idle) || exit 0  # 서버가 켜지는 중이면 그냥 둔다
echo "$r" | python3 -c 'import json,sys; d=json.load(sys.stdin); sys.exit(0 if d["idle_sec"] > 1800 and not d["busy"] else 1)' \
  && logger -t tapme "30분 동안 사용이 없어 서버를 끕니다" && /sbin/shutdown -h now
